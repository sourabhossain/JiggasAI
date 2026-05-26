import os
from django.http import FileResponse, Http404
from django.utils.decorators import method_decorator
from django.views.decorators.clickjacking import xframe_options_sameorigin
from django.views.generic import ListView
from django.views import View
from django.shortcuts import get_object_or_404, redirect, render
from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib.auth.models import User
from django.db.models import Sum
from django.utils.text import get_valid_filename
from .models import Document
from .forms import DocumentUploadForm
from agent.ingestor import ingest_document


class AdminRequiredMixin(UserPassesTestMixin):

    def test_func(self):
        return self.request.user.is_staff

    def handle_no_permission(self):
        if self.request.user.is_authenticated:
            from django.core.exceptions import PermissionDenied
            raise PermissionDenied
        from django.conf import settings
        return redirect(f"{settings.LOGIN_URL}?next={self.request.path}")


class DocumentListView(LoginRequiredMixin, AdminRequiredMixin, ListView):
    model = Document
    template_name = 'documents/list.html'
    context_object_name = 'documents'

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        agg = Document.objects.aggregate(total_chunks=Sum('chunk_count'))
        ctx['total_chunks'] = agg['total_chunks'] or 0
        ctx['total_users'] = User.objects.count()
        return ctx


class DocumentUploadView(LoginRequiredMixin, AdminRequiredMixin, View):
    template_name = 'documents/upload.html'

    def get(self, request):
        form = DocumentUploadForm()
        return render(request, self.template_name, {'form': form})

    def post(self, request):
        form = DocumentUploadForm(request.POST, request.FILES)

        if not form.is_valid():
            return render(request, self.template_name, {'form': form})

        file = form.cleaned_data['file']
        title = form.cleaned_data['title']
        if not title:
            raw_name = file.name.rsplit('.pdf', 1)[0]
            title = get_valid_filename(raw_name)[:200] or 'Untitled Document'

        doc = Document.objects.create(
            title=title,
            file=file,
            status='processing',
        )

        try:
            chunk_count = ingest_document(doc.file.path, doc_id=str(doc.id))
            doc.status = 'ready'
            doc.chunk_count = chunk_count
            doc.save()
            messages.success(
                request,
                f'"{title}" ingested successfully ({chunk_count} chunks).'
            )

        except ValueError as e:
            doc.status = 'failed'
            doc.error_message = str(e)
            doc.save()
            messages.error(request, f'Could not process "{title}": {e}')

        except Exception as e:
            print(f"[INGEST ERROR] doc_id={doc.id} error={type(e).__name__}: {e}")
            doc.status = 'failed'
            doc.error_message = f"{type(e).__name__}: {e}"
            doc.save()
            messages.error(
                request,
                f'Upload failed for "{title}". Please try again.'
            )

        return redirect('document-list')


class DocumentPreviewView(LoginRequiredMixin, AdminRequiredMixin, View):
    @method_decorator(xframe_options_sameorigin)
    def get(self, request, pk):
        doc = get_object_or_404(Document, pk=pk)

        if not doc.file:
            raise Http404("No file attached.")

        file_path = doc.file.path
        if not os.path.exists(file_path):
            raise Http404("File not found on disk.")

        response = FileResponse(open(file_path, "rb"), content_type="application/pdf")
        response["Content-Disposition"] = (
            f'inline; filename="{os.path.basename(file_path)}"'
        )
        return response


class DocumentDeleteView(LoginRequiredMixin, AdminRequiredMixin, View):

    def post(self, request, pk):
        doc = get_object_or_404(Document, pk=pk)
        title = doc.title
        errors = []

        try:
            from agent.chroma_client import delete_document_chunks
            deleted_count = delete_document_chunks(doc_id=str(doc.id))
            print(f"[DELETE] Removed {deleted_count} chunks from ChromaDB")
        except Exception as e:
            errors.append(f"ChromaDB error: {e}")
            print(f"[DELETE] ChromaDB failed: {e}")

        try:
            file_path = doc.file.path
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"[DELETE] Removed file: {file_path}")
        except Exception as e:
            errors.append(f"File error: {e}")
            print(f"[DELETE] File delete failed: {e}")

        if errors:
            messages.error(
                request,
                f'Could not fully delete "{title}": {", ".join(errors)}. Please try again.',
            )
        else:
            doc.delete()
            messages.success(request, f'"{title}" deleted successfully.')

        return redirect('document-list')
