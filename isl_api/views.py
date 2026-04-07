"""
HAATH — Complete views.py
Fixes: gesture prediction, SMS alerts, Kannada/Hindi TTS support
"""
import numpy as np
import pickle, json, os
from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings

# ── Serve pages ───────────────────────────────────────────
def index(request):
    html_path = os.path.join(settings.BASE_DIR, 'templates', 'index.html')
    with open(html_path, 'r', encoding='utf-8') as f:
        return HttpResponse(f.read(), content_type='text/html')

def call_page(request):
    html_path = os.path.join(settings.BASE_DIR, 'templates', 'call.html')
    with open(html_path, 'r', encoding='utf-8') as f:
        return HttpResponse(f.read(), content_type='text/html')

# ── Lazy load LSTM model ──────────────────────────────────
_model = _encoder = None

def get_model():
    global _model, _encoder
    if _model is None:
        try:
            import tensorflow as tf
            _model   = tf.keras.models.load_model(settings.LSTM_MODEL_PATH)
            with open(settings.LABEL_ENCODER_PATH, 'rb') as f:
                _encoder = pickle.load(f)
            print("✅ LSTM model loaded")
        except Exception as e:
            print(f"⚠️ Model load failed: {e}")
            return None, None
    return _model, _encoder

def normalize_frame(frame_63):
    """Normalize 63-float frame relative to wrist."""
    wx, wy, wz = frame_63[0], frame_63[1], frame_63[2]
    out = []
    for i in range(0, len(frame_63), 3):
        out.extend([frame_63[i]-wx, frame_63[i+1]-wy, frame_63[i+2]-wz])
    return out

# ── Gesture metadata ──────────────────────────────────────
GESTURE_INFO = {
    "HELLO":     {"emoji":"👋","category":"greeting",
                  "hindi":"नमस्ते","kannada":"ನಮಸ್ಕಾರ"},
    "THANK_YOU": {"emoji":"🙏","category":"courtesy",
                  "hindi":"धन्यवाद","kannada":"ಧನ್ಯವಾದ"},
    "SORRY":     {"emoji":"😔","category":"courtesy",
                  "hindi":"माफ करना","kannada":"ಕ್ಷಮಿಸಿ"},
    "PLEASE":    {"emoji":"🤲","category":"courtesy",
                  "hindi":"कृपया","kannada":"ದಯವಿಟ್ಟು"},
    "HELP":      {"emoji":"🆘","category":"urgent",
                  "hindi":"मदद करो","kannada":"ಸಹಾಯ ಮಾಡಿ"},
    "YES":       {"emoji":"✅","category":"response",
                  "hindi":"हाँ","kannada":"ಹೌದು"},
    "NO":        {"emoji":"❌","category":"response",
                  "hindi":"नहीं","kannada":"ಇಲ್ಲ"},
    "STOP":      {"emoji":"✋","category":"action",
                  "hindi":"रुको","kannada":"ನಿಲ್ಲಿ"},
    "COME":      {"emoji":"🫵","category":"action",
                  "hindi":"आओ","kannada":"ಬನ್ನಿ"},
    "GO":        {"emoji":"👉","category":"action",
                  "hindi":"जाओ","kannada":"ಹೋಗಿ"},
    "WATER":     {"emoji":"💧","category":"needs",
                  "hindi":"पानी","kannada":"ನೀರು"},
    "FOOD":      {"emoji":"🍽️","category":"needs",
                  "hindi":"खाना","kannada":"ಊಟ"},
    "DOCTOR":    {"emoji":"👨‍⚕️","category":"emergency",
                  "hindi":"डॉक्टर","kannada":"ವೈದ್ಯರು"},
    "WASHROOM":  {"emoji":"🚻","category":"needs",
                  "hindi":"शौचालय","kannada":"ಶೌಚಾಲಯ"},
    "EMERGENCY": {"emoji":"🚨","category":"emergency",
                  "hindi":"आपातकाल","kannada":"ತುರ್ತು"},
}

# ── SMS Alert ──────────────────────────────────────────────
def send_emergency_sms(gesture_name, confidence):
    """Send SMS via Twilio when emergency gesture detected."""
    try:
        from twilio.rest import Client
        account_sid = os.environ.get('TWILIO_ACCOUNT_SID', '')
        auth_token  = os.environ.get('TWILIO_AUTH_TOKEN', '')
        from_number = os.environ.get('TWILIO_FROM', '')
        to_number   = os.environ.get('EMERGENCY_CONTACT', '')

        if not all([account_sid, auth_token, from_number, to_number]):
            print("⚠️ Twilio credentials not set in environment variables")
            return False

        client = Client(account_sid, auth_token)
        info   = GESTURE_INFO.get(gesture_name, {})
        client.messages.create(
            body=(
                f"🚨 HAATH EMERGENCY ALERT\n"
                f"Gesture: {gesture_name} {info.get('emoji','')}\n"
                f"Confidence: {confidence:.0%}\n"
                f"A deaf user needs immediate assistance!\n"
                f"Please respond urgently."
            ),
            from_=from_number,
            to=to_number
        )
        print(f"✅ Emergency SMS sent for {gesture_name}")
        return True
    except ImportError:
        print("⚠️ twilio not installed: pip install twilio")
        return False
    except Exception as e:
        print(f"⚠️ SMS failed: {e}")
        return False

# Track last SMS time to avoid spam
_last_sms_time = {}

def should_send_sms(gesture_name):
    """Rate limit: send SMS max once per 60 seconds per gesture."""
    import time
    now  = time.time()
    last = _last_sms_time.get(gesture_name, 0)
    if now - last > 60:
        _last_sms_time[gesture_name] = now
        return True
    return False

def manifest(request):
    """Serves the PWA manifest file."""
    from django.http import JsonResponse
    return JsonResponse({
        "name": "Haath — ISL Communication",
        "short_name": "Haath",
        "description": "AI-Powered Indian Sign Language Communication Platform",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#040810",
        "theme_color": "#00e5a0",
        "orientation": "portrait",
        "icons": [
            {"src": "/static/icon-192.png", "sizes": "192x192", "type": "image/png"},
            {"src": "/static/icon-512.png", "sizes": "512x512", "type": "image/png"}
        ],
        "categories": ["medical", "accessibility", "communication"],
        "lang": "en-IN"
    })


# ── API Endpoints ──────────────────────────────────────────
@api_view(['GET'])
def health_check(request):
    model, _ = get_model()
    stats_path = os.path.join(settings.BASE_DIR, 'ml_model', 'training_stats.json')
    accuracy = None
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            accuracy = json.load(f).get('accuracy')
    return Response({
        "status":       "healthy",
        "model_loaded": model is not None,
        "model_type":   "LSTM Dynamic",
        "sequence_len": 30,
        "version":      "4.0.0",
        "gestures":     len(GESTURE_INFO),
        "accuracy":     accuracy,
        "features":     ["sms_alerts","hindi_tts","kannada_tts","subtitles"],
    })

@api_view(['GET'])
def list_gestures(request):
    return Response({
        "gestures": [
            {
                "name":     k,
                "display":  k.replace('_', ' '),
                "emoji":    v["emoji"],
                "category": v["category"],
                "hindi":    v.get("hindi", k),
                "kannada":  v.get("kannada", k),
            }
            for k, v in GESTURE_INFO.items()
        ],
        "total": len(GESTURE_INFO)
    })

@api_view(['GET'])
def model_info(request):
    stats_path = os.path.join(settings.BASE_DIR, 'ml_model', 'training_stats.json')
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            return Response(json.load(f))
    return Response({"error": "Model not trained yet"}, status=404)

@api_view(['POST'])
def predict_sequence(request):
    """
    Core prediction endpoint.
    Receives 30 frames, runs LSTM, returns prediction + multilingual text.

    Request: { "sequence": [[63 floats], ...30 frames] }
    Response: { "gesture": "HELLO", "confidence": 0.94, "hindi": "नमस्ते", ... }
    """
    model, encoder = get_model()
    if model is None:
        return Response(
            {"error": "Model not loaded. Run: python ml_model/train_lstm.py"},
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )

    sequence = request.data.get('sequence')
    if not sequence:
        return Response({"error": "No sequence provided"}, status=400)

    if len(sequence) != 30:
        return Response({
            "error": f"Expected 30 frames, got {len(sequence)}"
        }, status=400)

    try:
        # Normalize each frame
        normalized = []
        for frame in sequence:
            if len(frame) != 63:
                return Response({"error": "Each frame must have 63 values"}, status=400)
            normalized.append(normalize_frame(frame))

        # Shape: (1, 30, 63)
        X        = np.array([normalized], dtype=np.float32)
        proba    = model.predict(X, verbose=0)[0]
        pred_idx = int(np.argmax(proba))
        gesture  = encoder.inverse_transform([pred_idx])[0]
        conf     = float(proba[pred_idx])

        # Top 3 predictions
        top3 = sorted(
            [{"gesture": encoder.inverse_transform([i])[0],
              "confidence": float(proba[i])}
             for i in range(len(proba))],
            key=lambda x: x["confidence"], reverse=True
        )[:3]

        info = GESTURE_INFO.get(gesture, {})
        is_emergency = info.get("category") == "emergency"

        # Send SMS for emergency gestures
        if is_emergency and conf > 0.80 and should_send_sms(gesture):
            send_emergency_sms(gesture, conf)

        return Response({
            "gesture":         gesture,
            "display":         gesture.replace('_', ' '),
            "confidence":      conf,
            "emoji":           info.get("emoji", "✋"),
            "category":        info.get("category", "action"),
            "hindi":           info.get("hindi", gesture),
            "kannada":         info.get("kannada", gesture),
            "is_emergency":    is_emergency,
            "top_predictions": top3,
        })

    except Exception as e:
        return Response({"error": str(e)}, status=500)
