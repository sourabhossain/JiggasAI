from django import forms
from django.core.exceptions import ValidationError


MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


class DocumentUploadForm(forms.Form):
    title = forms.CharField(
        max_length=255,
        required=False,
        widget=forms.TextInput(attrs={
            "placeholder": "e.g. Machine Learning Notes",
            "class": "w-full px-3.5 py-2.5 rounded-xl border border-gray-200 text-sm focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent transition",
        })
    )
    file = forms.FileField(
        widget=forms.FileInput(attrs={
            "accept": ".pdf",
            "class": "hidden",
            "id": "file-input",
        })
    )

    def clean_file(self):
        file = self.cleaned_data.get("file")
        if not file:
            raise ValidationError("Please select a file.")

        if file.size == 0:
            raise ValidationError("The uploaded file is empty.")

        if file.size > MAX_FILE_SIZE_BYTES:
            raise ValidationError(
                f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB. "
                f"Your file is {file.size // (1024 * 1024)}MB."
            )

        if not file.name.lower().endswith(".pdf"):
            ext = file.name.rsplit(".", 1)[-1].upper() if "." in file.name else "(no extension)"
            raise ValidationError(
                f"Only PDF files are supported. You uploaded: {ext}"
            )

        file.seek(0)
        header = file.read(5)
        file.seek(0)
        if header != b"%PDF-":
            raise ValidationError(
                "File does not appear to be a valid PDF. "
                "Please upload a real PDF file."
            )

        return file

    def clean_title(self):
        return self.cleaned_data.get("title", "").strip()
