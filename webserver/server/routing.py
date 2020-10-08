# Another file for Channels...

from django.urls import re_path

from . import consumers

# need to use regex path due in limitations of URLRouter
websocket_urlpatterns = [
    re_path(r'ws/users/(?P<username>\w+)/records/$', consumers.EchoConsumer),
    re_path(r'ws/users/(?P<username>\w+)/process/$', consumers.FrameProcessConsumer),
]