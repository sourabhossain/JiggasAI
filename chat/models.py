from django.db import models


class ChatSession(models.Model):
    session_key = models.CharField(max_length=100, unique=True, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    @property
    def first_message(self):
        msg = self.messages.filter(role='user').first()
        if msg:
            text = msg.content
            return text[:40] + ('...' if len(text) > 40 else '')
        return 'New chat'


class Message(models.Model):
    ROLES = [('user', 'User'), ('assistant', 'Assistant')]

    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=10, choices=ROLES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']
