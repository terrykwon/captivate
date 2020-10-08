from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack

import server.routing

''' Upon connection, the ProtocolTypeRouter will inspect the type
    of connection (ws:// or http://). If it is ws, then the connection 
    is passed to the AuthMiddlewareStack. Then URLRouter will examine
    the url and route it to the proper consumer.
'''
application = ProtocolTypeRouter({
    # (http -> django views is added by default)
    'websocket': AuthMiddlewareStack(
        URLRouter(
            server.routing.websocket_urlpatterns
        )
    ),
})