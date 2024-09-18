let extractedData = null;

async function processInvoice() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please upload an invoice.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/process-invoice/', {
        method: 'POST',
        body: formData
    });

    if (response.ok) {
        const result = await response.json();
        extractedData = result.data;
        displayData(extractedData);
        document.getElementById('outputContainer').style.display = 'block';
    } else {
        alert('Failed to process the invoice.');
    }
}

function displayData(data) {
    let tableHtml = '<table class="table"><thead><tr><th>Field</th><th>Value</th></tr></thead><tbody>';
    data.forEach(item => {
        tableHtml += `<tr><td>${item.Field}</td><td>${item.Value}</td></tr>`;
    });
    tableHtml += '</tbody></table>';
    document.getElementById('outputTable').innerHTML = tableHtml;
}

function downloadData(type) {
    if (!extractedData) {
        alert('No data to download');
        return;
    }

    if (type === 'json') {
        const jsonStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(extractedData));
        const downloadAnchor = document.createElement('a');
        downloadAnchor.setAttribute("href", jsonStr);
        downloadAnchor.setAttribute("download", "invoice_data.json");
        downloadAnchor.click();
    } else if (type === 'excel') {
        const formData = new FormData();
        formData.append('file', document.getElementById('fileInput').files[0]);
        
        fetch('/download-invoice-excel/', {
            method: 'POST',
            body: formData
        }).then(response => response.blob())
          .then(blob => {
              const url = window.URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.href = url;
              a.download = "invoice_details.xlsx";
              document.body.appendChild(a);
              a.click();
              a.remove();
          });
    }
}
