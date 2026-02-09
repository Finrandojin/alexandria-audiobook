/**
 * Component Loader for Alexandria Audiobook
 * Dynamically loads HTML components into the page
 */

(function() {
    'use strict';

    // Component definitions
    const components = [
        { name: 'setup', target: 'setup-container' },
        { name: 'script', target: 'script-container' },
        { name: 'voices', target: 'voices-container' },
        { name: 'editor', target: 'editor-container' },
        { name: 'audio', target: 'audio-container' }
    ];

    /**
     * Load a single component
     * @param {Object} component - Component configuration
     * @returns {Promise} Promise that resolves when component is loaded
     */
    async function loadComponent(component) {
        try {
            const response = await fetch(`/static/components/${component.name}.html`);
            if (!response.ok) {
                throw new Error(`Failed to load ${component.name}: ${response.status}`);
            }
            const html = await response.text();
            const container = document.getElementById(component.target);
            if (container) {
                container.innerHTML = html;
            } else {
                console.warn(`Container ${component.target} not found for component ${component.name}`);
            }
        } catch (error) {
            console.error(`Error loading component ${component.name}:`, error);
            throw error;
        }
    }

    /**
     * Load all components in parallel
     * @returns {Promise} Promise that resolves when all components are loaded
     */
    async function loadAllComponents() {
        try {
            await Promise.all(components.map(component => loadComponent(component)));
            console.log('All components loaded successfully');
            
            // Dispatch custom event to notify that components are ready
            window.dispatchEvent(new CustomEvent('componentsLoaded'));
        } catch (error) {
            console.error('Error loading components:', error);
        }
    }

    // Load components when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', loadAllComponents);
    } else {
        loadAllComponents();
    }
})();
