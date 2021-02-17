let lock = false;

function createSampleProject(e) {
    e.preventDefault();
    if (lock) {
        return;
    }
    lock = true;
    const projectName = document.getElementById('projectName').value;
    const file = document.getElementById('file').value;
    const projectDes = document.getElementById('projectDes').value;

    document.getElementById('notify').innerHTML = 'Processing...';

    fetch('/api/create', {
        method: 'POST',
        cache: 'no-cache',
        body: JSON.stringify({
            projectName,
            projectDes,
            file,
        }),
    })
        .then((response) => response.json())
        .then((data) => {
            if ('error' in data) {
                throw new Error(data.error);
            }
            lock = false;
            document.getElementById('notify').innerHTML = `OK. Link <a href="${data['link']}">${data['link']}</a>`;
        })
        .catch((error) => {
            lock = false;
            document.getElementById('notify').innerHTML = error;
            console.log(error);
        })
}

document.getElementById('main-form').addEventListener('submit', createSampleProject, true);