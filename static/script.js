// Script to enhance user interaction on file upload

document.addEventListener('DOMContentLoaded', function () {
    const fileInput = document.querySelector('input[type="file"]');
    const form = document.querySelector('form');

    fileInput.addEventListener('change', function () {
        const fileName = fileInput.files[0].name;
        alert(`File selected: ${fileName}`);
    });

    form.addEventListener('submit', function () {
        alert('Uploading the invoice, please wait...');
    });
});
