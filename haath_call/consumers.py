"""
╔══════════════════════════════════════════════════════════════╗
║  HAATH CALL — WebSocket Signaling Server                     ║
║  Handles WebRTC peer connection negotiation                  ║
╚══════════════════════════════════════════════════════════════╝

Flow:
  1. User A creates a room → gets a room_code
  2. User B joins with that room_code
  3. This consumer relays WebRTC offer/answer/ICE between them
  4. Once connected, video flows peer-to-peer (not through server)
"""

import json
import logging
from channels.generic.websocket import AsyncWebsocketConsumer

logger = logging.getLogger(__name__)

# In-memory room registry
# { room_code: { "host": channel_name, "guest": channel_name } }
ROOMS = {}


class CallConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        self.room_code   = self.scope['url_route']['kwargs']['room_code']
        self.room_group  = f"call_{self.room_code}"
        self.role        = None   # "host" or "guest"

        # Join the channel group
        await self.channel_layer.group_add(
            self.room_group,
            self.channel_name
        )
        await self.accept()

        # Register in room
        if self.room_code not in ROOMS:
            ROOMS[self.room_code] = {}

        room = ROOMS[self.room_code]

        if 'host' not in room:
            room['host'] = self.channel_name
            self.role    = 'host'
            await self.send(json.dumps({
                'type': 'role',
                'role': 'host',
                'room': self.room_code,
                'message': 'Room created. Share the room code with the other person.'
            }))
            logger.info(f"Host joined room {self.room_code}")

        elif 'guest' not in room:
            room['guest'] = self.channel_name
            self.role     = 'guest'
            await self.send(json.dumps({
                'type': 'role',
                'role': 'guest',
                'room': self.room_code,
                'message': 'Joined room. Connecting...'
            }))
            # Notify host that guest has arrived
            await self.channel_layer.group_send(
                self.room_group,
                {
                    'type':    'peer_message',
                    'payload': {'type': 'guest_joined'},
                    'sender':  self.channel_name,
                }
            )
            logger.info(f"Guest joined room {self.room_code}")

        else:
            # Room is full
            await self.send(json.dumps({
                'type':    'error',
                'message': 'Room is full. Please create a new room.'
            }))
            await self.close()

    async def disconnect(self, close_code):
        room = ROOMS.get(self.room_code, {})

        # Remove from room
        if room.get('host') == self.channel_name:
            room.pop('host', None)
        elif room.get('guest') == self.channel_name:
            room.pop('guest', None)

        # Clean up empty room
        if not room:
            ROOMS.pop(self.room_code, None)

        # Notify the other peer
        await self.channel_layer.group_send(
            self.room_group,
            {
                'type':    'peer_message',
                'payload': {'type': 'peer_disconnected'},
                'sender':  self.channel_name,
            }
        )

        await self.channel_layer.group_discard(
            self.room_group,
            self.channel_name
        )
        logger.info(f"Peer left room {self.room_code}")

    async def receive(self, text_data):
        """Relay all WebRTC signaling messages to the other peer."""
        try:
            data = json.loads(text_data)
        except json.JSONDecodeError:
            return

        # Relay to everyone else in the group
        await self.channel_layer.group_send(
            self.room_group,
            {
                'type':    'peer_message',
                'payload': data,
                'sender':  self.channel_name,
            }
        )

    async def peer_message(self, event):
        """Send message to this WebSocket (only if not the sender)."""
        if event['sender'] != self.channel_name:
            await self.send(json.dumps(event['payload']))
