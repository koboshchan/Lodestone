// Entry point. The bundle is injected at the end of <body>, so the DOM already exists
// and nothing here waits for DOMContentLoaded — same as the original inline script.
//
// Listener registration is explicit rather than a side effect of importing each module.
// Import order under a bundler is an implementation detail, and wiring that depended on
// it would be wiring that breaks when an import moves.

import { IS_MAC } from './constants.js';
import { modKeyHint, undoShortcutHint, redoShortcutHint } from './dom.js';
import { resetHistory } from './history.js';
import { registerImageListeners } from './image.js';
import { registerInteractionListeners } from './interactions.js';
import { registerMenuListeners } from './menu.js';
import { loadStateFromStorage, registerPersistenceListeners } from './persistence.js';

registerImageListeners();
registerPersistenceListeners();
registerInteractionListeners();
registerMenuListeners();

if (IS_MAC) modKeyHint.textContent = 'Cmd';
undoShortcutHint.textContent = IS_MAC ? '⌘Z' : 'Ctrl+Z';
redoShortcutHint.textContent = IS_MAC ? '⇧⌘Z' : 'Ctrl+Shift+Z';

// Initialize & restore session on startup
loadStateFromStorage();

// The restored session is the baseline, not a state to be undone past.
resetHistory();
