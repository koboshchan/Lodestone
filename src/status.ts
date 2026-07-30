import { statusBadge } from './dom.js';

export function updateStatus(msg: string): void {
  statusBadge.textContent = msg;
}
