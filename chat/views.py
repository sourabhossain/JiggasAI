import json
import re
from django.shortcuts import render, get_object_or_404
from django.http import StreamingHttpResponse, HttpResponseBadRequest, JsonResponse
from django.views import View
from django.contrib.auth.mixins import LoginRequiredMixin
from .models import ChatSession, Message
from agent.rag_agent import run_agent, run_agent_stream


_URL_RE = re.compile(r"^https?://[^\s<>\"']+$")


def sanitize_source(source: str) -> str:
    if not source or not isinstance(source, str):
        return ""
    source = source.strip()
    if source.startswith(("http://", "https://")):
        if _URL_RE.match(source):
            return source[:200]
        return ""
    source = re.sub(r"[^\w\s\-.]", "", source)
    return source[:100].strip()


class ChatView(LoginRequiredMixin, View):
    def get(self, request, pk=None):
        if pk:
            # Sidebar link clicked — load that specific session
            session = get_object_or_404(ChatSession, pk=pk)
            request.session['chat_session_id'] = session.id
        elif request.GET.get('new'):
            # "New chat" button — always create a fresh session
            session = ChatSession.objects.create()
            request.session['chat_session_id'] = session.id
        else:
            # Normal load — restore last session or create one
            session_id = request.session.get('chat_session_id')
            session = ChatSession.objects.filter(id=session_id).first() if session_id else None
            if not session:
                session = ChatSession.objects.create()
                request.session['chat_session_id'] = session.id

        recent_sessions = ChatSession.objects.order_by('-created_at')[:10]
        return render(request, 'chat/chat.html', {
            'session': session,
            'messages': session.messages.all(),
            'recent_sessions': recent_sessions,
            'current_session_id': session.id,
        })


class ChatAskView(LoginRequiredMixin, View):
    def post(self, request):
        data = json.loads(request.body)
        query = data.get('query', '').strip()
        session_id = data.get('session_id')

        session = get_object_or_404(ChatSession, id=session_id)

        previous = list(session.messages.all())
        history = []
        i = 0
        while i < len(previous) - 1:
            if previous[i].role == 'user' and previous[i + 1].role == 'assistant':
                history.append({
                    "user": previous[i].content,
                    "assistant": previous[i + 1].content,
                })
                i += 2
            else:
                i += 1

        Message.objects.create(session=session, role='user', content=query)

        result = run_agent(query=query, history=history)
        answer = result['answer']

        Message.objects.create(session=session, role='assistant', content=answer)

        return render(request, 'chat/partials/message.html', {
            'user_message': query,
            'assistant_message': answer,
        })


class ChatStreamView(LoginRequiredMixin, View):
    def get(self, request):
        query = request.GET.get('query', '').strip()

        session_id = request.session.get('chat_session_id')
        session = ChatSession.objects.filter(id=session_id).first() if session_id else None
        if not session:
            return HttpResponseBadRequest("No active session")

        if not query:
            return HttpResponseBadRequest("Missing query")

        previous = list(session.messages.all())
        history = []
        i = 0
        while i < len(previous) - 1:
            if previous[i].role == 'user' and previous[i + 1].role == 'assistant':
                history.append({
                    "user": previous[i].content,
                    "assistant": previous[i + 1].content,
                })
                i += 2
            else:
                i += 1

        Message.objects.create(session=session, role='user', content=query)

        def event_stream():
            stored_sources = []
            for token in run_agent_stream(query=query, history=history):
                if token.startswith("__ANSWER__"):
                    full_answer = token[len("__ANSWER__"):]
                    Message.objects.create(session=session, role='assistant', content=full_answer)
                    safe_sources = [sanitize_source(s) for s in stored_sources]
                    safe_sources = [s for s in safe_sources if s]
                    yield f"data: {json.dumps({'done': True, 'sources': safe_sources})}\n\n"
                elif token.startswith("__SOURCES__"):
                    raw = token[len("__SOURCES__"):].split(',')
                    stored_sources = [s.strip() for s in raw if s.strip()]
                elif token:
                    yield f"data: {json.dumps({'token': token})}\n\n"

        response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
        response['Cache-Control'] = 'no-cache'
        response['X-Accel-Buffering'] = 'no'
        return response


class ChatSessionDeleteView(LoginRequiredMixin, View):
    def post(self, request, pk):
        session = get_object_or_404(ChatSession, pk=pk)
        session.delete()
        return JsonResponse({'ok': True})
