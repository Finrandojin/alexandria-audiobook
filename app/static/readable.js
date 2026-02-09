/**
 * Loads a JSON script file and displays it in a human redable format in a new window
 * @param {string} filename - Path to the JSON file to load
 */
function displayReadableScript(filename) {
  fetch(filename)
    .then(response => {
      if (!response.ok) {
        throw new Error(`Failed to load file: ${response.statusText}`);
      }
      return response.json();
    })
    .then(data => {
      const newWindow = window.open('', '_blank', 'width=1000,height=800');
      
      if (!newWindow) {
        alert('Please allow popups to view the formatted script');
        return;
      }

      const html = generateReadablePage(data);
      newWindow.document.write(html);
      newWindow.document.close();
    })
    .catch(error => {
      alert(`Error loading file: ${error.message}`);
      console.error(error);
    });
}

/**
 * Generates the HTML content for the readable page
 * @param {Array} entries - Array of script entries
 * @returns {string} Complete HTML page as string
 */
function generateReadablePage(entries) {
  const styles = `
    body {
      font-family: Georgia, 'Times New Roman', serif;
      background-color: white;
      color: black;
      margin: 20px;
      padding: 20px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
    }
    td {
      padding: 15px 10px;
      vertical-align: top;
    }
    .speaker-col {
      width: 16.67%;
      font-weight: bold;
    }
    .content-col {
      width: 83.33%;
    }
    .text {
      font-size: 16px;
      margin-bottom: 8px;
      line-height: 1.6;
    }
    .instruct {
      font-size: 12px;
      text-align: right;
      color: #333;
      font-style: italic;
    }
  `;

  let tableRows = '';
  entries.forEach(entry => {
    const speaker = entry.speaker || '';
    const text = entry.text || '';
    const instruct = entry.instruct || '';

    tableRows += `
      <tr>
        <td class="speaker-col">${escapeHtml(speaker)}</td>
        <td class="content-col">
          <div class="text">${escapeHtml(text)}</div>
          <div class="instruct">${escapeHtml(instruct)}</div>
        </td>
      </tr>
    `;
  });

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Script Reader</title>
  <style>${styles}</style>
</head>
<body>
  <table>
    ${tableRows}
  </table>
</body>
</html>`;
}

/**
 * Output sanitising function to prevent XSS attacks by escaping HTML special characters
 * @param {string} text - Text to escape
 * @returns {string} Escaped text
 */
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
