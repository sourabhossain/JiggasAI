from django.urls import path
from .views import (
    DocumentListView,
    DocumentUploadView,
    DocumentDeleteView,
    DocumentPreviewView,
)

urlpatterns = [
    path('', DocumentListView.as_view(), name='document-list'),
    path('upload/', DocumentUploadView.as_view(), name='document-upload'),
    path('<int:pk>/delete/', DocumentDeleteView.as_view(), name='document-delete'),
    path('<int:pk>/preview/', DocumentPreviewView.as_view(), name='document-preview'),
]
