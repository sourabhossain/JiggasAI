from django.http import HttpResponse


def home(request):
    return HttpResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>JiggasAI</title>
    <style>
        body { font-family: sans-serif; max-width: 600px; margin: 60px auto; padding: 0 20px; }
        h1 { color: #333; }
        .status { padding: 10px; border-radius: 6px; margin: 8px 0; background: #e8f5e9; }
        a { color: #1a73e8; }
    </style>
</head>
<body>
    <h1>JiggasAI</h1>
    <p>Your personal RAG agent is up and running.</p>
    <div class="status">Django &mdash; running</div>
    <div class="status">MySQL &mdash; connected (migrations applied)</div>
    <div class="status">ChromaDB &mdash; <a href="http://localhost:8001/api/v1/heartbeat" target="_blank">check heartbeat</a></div>
    <p><a href="/admin/">Django Admin</a></p>
</body>
</html>
""")
