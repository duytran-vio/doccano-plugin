let lock = false;
let transform = {
    "<>": "tr",
    "children": [
        { "<>": "td", "html": "${category}" },
        { "<>": "td", "html": "${precision}" },
        { "<>": "td", "html": "${recall}" },
        { "<>": "td", "html": "${f1-score}" },
        { "<>": "td", "html": "${support}" },
    ],
};
let header = `
    <th>Category</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-score</th>
    <th>Support</th>
`

function createSampleProject(e) {
    e.preventDefault();
    if (lock) {
        return;
    }
    lock = true;
    const projectId = document.getElementById('projectId').value;
    const initialProjectId = document.getElementById('initialProjectId').value;

    document.getElementById('notify').innerHTML = 'Processing...';

    fetch('/api/evaluate?' + new URLSearchParams({
        projectId,
    }), {
        method: 'GET',
        cache: 'no-cache'
    })
        .then((response) => {
            if (response.status === 500) {
                return response.json().then((data) => {
                    throw new Error(data['error']);
                });
            }
            return response.blob();
        })
        .then((blob) => {
            lock = false;
            document.getElementById('notify').innerHTML = 'Done';

            let url = window.URL.createObjectURL(blob);
            let a = document.createElement('a');
            a.href = url;
            a.download = `Tesing_${initialProjectId}_Sample_${projectId}.xlsx`;
            document.body.appendChild(a);
            a.click();
            a.remove();
        })
        .catch((error) => {
            lock = false;
            document.getElementById('notify').innerHTML = error;
            document.getElementById('result').innerHTML = '';
            console.log(error);
        })
}

document.getElementById('main-form').addEventListener('submit', createSampleProject, true);