document.addEventListener('DOMContentLoaded', function () {
    const fileInput = document.querySelector('input[type="file"]');
    const submitButton = document.querySelector('button[type="submit"]');

    fileInput.addEventListener('change', function () {
        const fileName = fileInput.files[0].name;
        submitButton.textContent = `Upload ${fileName}`;
        submitButton.disabled = false;
    });
});

function resetForm() {
    document.getElementById('invoiceForm').reset();
    document.querySelector('button[type="submit"]').textContent = "Upload Invoice";
}
