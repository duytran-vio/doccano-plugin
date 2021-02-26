let lock = false;

function createSampleProject(e) {
    e.preventDefault();
    if (lock) {
        return;
    }
    lock = true;
    const projectId = document.getElementById('projectId').value;
    const start = document.getElementById('start').value;
    const end = document.getElementById('end').value;

    document.getElementById('notify').innerHTML = 'Processing...';

    fetch('/api/summary?' + new URLSearchParams({
        projectId, start, end
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
            a.download = `${projectId}-${start}-${end}_summary.xlsx`;
            document.body.appendChild(a);
            a.click();
            a.remove();
        })
        .catch((error) => {
            lock = false;
            document.getElementById('notify').innerHTML = error;
            console.log(error);
        })

    
}

document.getElementById('main-form').addEventListener('submit', createSampleProject, true);