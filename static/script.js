document.addEventListener('DOMContentLoaded', function () {
    const fileInput = document.querySelector('input[type="file"]');
    const submitButton = document.querySelector('button[type="submit"]');
    const resetButton = document.querySelector('button[type="reset"]');

    // Update the button text when a file is chosen
    fileInput.addEventListener('change', function () {
        const fileName = fileInput.files[0].name;
        submitButton.textContent = `Upload ${fileName}`;
        submitButton.disabled = false;
    });

    // Reset the form when reset button is clicked
    resetButton.addEventListener('click', function () {
        submitButton.textContent = 'Upload Invoice';
        submitButton.disabled = true;
        fileInput.value = '';  // Clear the file input field
    });
});