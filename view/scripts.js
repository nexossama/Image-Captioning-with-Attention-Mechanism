document.addEventListener('DOMContentLoaded', function () {
    const input = document.getElementById('images');
    const imagePreview = document.getElementById('imagePreview');
    const form = document.getElementById('storyForm');
    const generateButton = document.getElementById('generateStory');
    const generatedStory = document.getElementById('generatedStory');
    
    // Initialize FormData object
    const formData = new FormData();

    input.addEventListener('change', updateImagePreview);

    function updateImagePreview(event) {
        const files = event.target.files;
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const reader = new FileReader();
            reader.onload = function(e) {
                const img = document.createElement('img');
                img.src = e.target.result;
                img.classList.add('max-w-full', 'rounded-md', 'mb-2'); // Added margin-bottom for spacing
                const fileName = document.createElement('p');
                fileName.textContent = file.name;
                const container = document.createElement('div');
                container.classList.add('mb-4'); // Added margin-bottom for spacing
                container.appendChild(img);
                container.appendChild(fileName);
                imagePreview.appendChild(container);
            }
            reader.readAsDataURL(file);

            // Append each file to FormData object
            formData.append('files', file);
        }
    }

    generateButton.addEventListener('click', function () {
        const storyTypeInput = document.getElementById('storyType');

        if (formData.getAll('files').length === 0) {
            alert('Please upload at least one image.');
            return;
        }

        formData.append('story_type', storyTypeInput.value);

        fetch('http://0.0.0.0:5000/api/v1/generate/story', {  // Replace with your FastAPI endpoint
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.captions && data.story) {
                let captionsHtml = '<h3 class="text-xl font-semibold">Captions:</h3><ul>';
                data.captions.forEach(caption => {
                    captionsHtml += `<li>${caption}</li>`;
                });
                captionsHtml += '</ul>';
                
                const storyHtml = `<h3 class="text-xl font-semibold">Story:</h3><p>${data.story}</p>`;
                
                generatedStory.innerHTML = captionsHtml + storyHtml;
            } else {
                generatedStory.innerHTML = '<p>Failed to generate story. Please try again.</p>';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            generatedStory.innerHTML = '<p>There was an error processing your request. Please try again later.</p>';
        });
    });

    const nightModeToggle = document.getElementById('nightModeToggle');
    const body = document.body;

    nightModeToggle.addEventListener('click', () => {
        body.classList.toggle('night-mode');
    });
});
