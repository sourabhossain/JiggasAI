from django.urls import path
from .views import ChatView, ChatAskView, ChatStreamView, ChatSessionDeleteView

urlpatterns = [
    path('', ChatView.as_view(), name='chat'),
    path('<int:pk>/', ChatView.as_view(), name='chat_session'),
    path('<int:pk>/delete/', ChatSessionDeleteView.as_view(), name='chat-session-delete'),
    path('ask/', ChatAskView.as_view(), name='chat-ask'),
    path('stream/', ChatStreamView.as_view(), name='chat-stream'),
]
