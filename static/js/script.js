document.addEventListener('DOMContentLoaded', function() {
    // File input preview functionality
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (file) {
                // Check if file is an image
                if (!file.type.match('image.*')) {
                    alert('Please select an image file (jpg, jpeg, png)');
                    fileInput.value = '';
                    return;
                }
                
                // Check file size (limit to 5MB)
                if (file.size > 5 * 1024 * 1024) {
                    alert('File size should be less than 5MB');
                    fileInput.value = '';
                    return;
                }
                
                // If a preview element already exists, create it
                let previewContainer = document.getElementById('image-preview-container');
                if (!previewContainer) {
                    previewContainer = document.createElement('div');
                    previewContainer.id = 'image-preview-container';
                    previewContainer.className = 'mt-3';
                    fileInput.parentNode.appendChild(previewContainer);
                }
                
                // Clear previous preview
                previewContainer.innerHTML = '';
                
                // Create preview heading
                const previewHeading = document.createElement('h5');
                previewHeading.textContent = 'Image Preview:';
                previewContainer.appendChild(previewHeading);
                
                // Create image preview
                const img = document.createElement('img');
                img.className = 'img-fluid rounded mt-2';
                img.style.maxHeight = '200px';
                previewContainer.appendChild(img);
                
                // Read and display the image
                const reader = new FileReader();
                reader.onload = function(e) {
                    img.src = e.target.result;
                };
                reader.readAsDataURL(file);
            }
        });
    }
    
    // Form validation
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', function(event) {
            if (!fileInput.files.length) {
                event.preventDefault();
                alert('Please select an image to upload');
                return false;
            }
            
            // Show loading state
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
            }
            
            return true;
        });
    }
    
    // Ingredients input enhancement
    const ingredientsInput = document.getElementById('ingredients');
    if (ingredientsInput) {
        // Add placeholder text that disappears on focus
        ingredientsInput.addEventListener('focus', function() {
            if (this.placeholder === 'e.g., onion, garlic, rice, olive oil') {
                this.placeholder = '';
            }
        });
        
        ingredientsInput.addEventListener('blur', function() {
            if (this.value === '') {
                this.placeholder = 'e.g., onion, garlic, rice, olive oil';
            }
        });
        
        // Auto-format commas
        ingredientsInput.addEventListener('input', function() {
            // Replace multiple commas with a single comma
            this.value = this.value.replace(/,+/g, ',');
            // Remove commas at the beginning
            this.value = this.value.replace(/^,/, '');
        });
    }
    
    // Add smooth scrolling for result page
    if (window.location.pathname.includes('result')) {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
        
        // Add animation to recipe cards
        const recipeCards = document.querySelectorAll('.card');
        recipeCards.forEach((card, index) => {
            setTimeout(() => {
                card.style.transition = 'opacity 0.5s, transform 0.5s';
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, 100 * index);
        });
    }
});
