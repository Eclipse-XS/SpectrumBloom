import { ChangeDetectionStrategy, Component, input } from '@angular/core';

export type IconName =
  | 'grid'
  | 'table'
  | 'chart'
  | 'spark'
  | 'flask'
  | 'download'
  | 'settings'
  | 'menu'
  | 'search'
  | 'chevron'
  | 'more'
  | 'upload'
  | 'check'
  | 'warning'
  | 'database'
  | 'moon'
  | 'sun'
  | 'expand'
  | 'filter'
  | 'columns'
  | 'send'
  | 'code'
  | 'copy'
  | 'refresh'
  | 'close'
  | 'file'
  | 'info'
  | 'plus'
  | 'external'
  | 'eye'
  | 'clock'
  | 'trash'
  | 'lock';

@Component({
  selector: 'sb-svg-icon',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  styles: [
    `
      :host {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        line-height: 0;
        flex-shrink: 0;
      }
    `,
  ],
  template: `
    <svg
      aria-hidden="true"
      [attr.width]="size()"
      [attr.height]="size()"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="1.8"
      stroke-linecap="round"
      stroke-linejoin="round"
    >
      @switch (name()) {
        @case ('grid') {
          <rect x="3" y="3" width="7" height="7" rx="1" />
          <rect x="14" y="3" width="7" height="7" rx="1" />
          <rect x="3" y="14" width="7" height="7" rx="1" />
          <rect x="14" y="14" width="7" height="7" rx="1" />
        }
        @case ('table') {
          <rect x="3" y="4" width="18" height="16" rx="2" />
          <path d="M3 10h18M9 4v16" />
        }
        @case ('chart') {
          <path d="M4 19V9M10 19V4M16 19v-7M22 19H2" />
        }
        @case ('spark') {
          <path
            d="m12 3 1.3 4.2a5 5 0 0 0 3.4 3.4L21 12l-4.3 1.4a5 5 0 0 0-3.4 3.4L12 21l-1.3-4.2a5 5 0 0 0-3.4-3.4L3 12l4.3-1.4a5 5 0 0 0 3.4-3.4L12 3Z"
          />
        }
        @case ('flask') {
          <path d="M9 3h6M10 3v5l-5.6 9.2A2.5 2.5 0 0 0 6.6 21h10.8a2.5 2.5 0 0 0 2.2-3.8L14 8V3" />
          <path d="M7.5 15h9" />
        }
        @case ('download') {
          <path d="M12 3v12m0 0 4-4m-4 4-4-4" />
          <path d="M4 19v2h16v-2" />
        }
        @case ('settings') {
          <circle cx="12" cy="12" r="3" />
          <path
            d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-2.8 2.8-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6v.2h-4V21a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1L4.2 17l.1-.1a1.7 1.7 0 0 0 .3-1.9A1.7 1.7 0 0 0 3 14H2.8v-4H3a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9L4.2 7 7 4.2l.1.1a1.7 1.7 0 0 0 1.9.3A1.7 1.7 0 0 0 10 3V2.8h4V3a1.7 1.7 0 0 0 1 1.6 1.7 1.7 0 0 0 1.9-.3l.1-.1L19.8 7l-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.6 1h.2v4H21a1.7 1.7 0 0 0-1.6 1Z"
          />
        }
        @case ('menu') {
          <path d="M4 7h16M4 12h16M4 17h16" />
        }
        @case ('search') {
          <circle cx="11" cy="11" r="7" />
          <path d="m20 20-4-4" />
        }
        @case ('chevron') {
          <path d="m9 18 6-6-6-6" />
        }
        @case ('more') {
          <circle cx="5" cy="12" r="1" />
          <circle cx="12" cy="12" r="1" />
          <circle cx="19" cy="12" r="1" />
        }
        @case ('upload') {
          <path d="M12 16V4m0 0L7 9m5-5 5 5" />
          <path d="M4 15v5h16v-5" />
        }
        @case ('check') {
          <path d="m5 12 4 4L19 6" />
        }
        @case ('warning') {
          <path d="M12 3 2.8 20h18.4L12 3Z" />
          <path d="M12 9v4m0 3h.01" />
        }
        @case ('database') {
          <ellipse cx="12" cy="5" rx="8" ry="3" />
          <path d="M4 5v7c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12v7c0 1.7 3.6 3 8 3s8-1.3 8-3v-7" />
        }
        @case ('moon') {
          <path d="M20 15.5A8.5 8.5 0 0 1 8.5 4 8.5 8.5 0 1 0 20 15.5Z" />
        }
        @case ('sun') {
          <circle cx="12" cy="12" r="4" />
          <path
            d="M12 2v2m0 16v2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M2 12h2m16 0h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"
          />
        }
        @case ('expand') {
          <path d="M8 3H3v5M16 3h5v5M8 21H3v-5M16 21h5v-5" />
        }
        @case ('filter') {
          <path d="M3 5h18l-7 8v6l-4 2v-8L3 5Z" />
        }
        @case ('columns') {
          <rect x="3" y="4" width="18" height="16" rx="2" />
          <path d="M9 4v16m6-16v16" />
        }
        @case ('send') {
          <path d="m3 11 18-8-8 18-2-8-8-2Zm8 2 10-10" />
        }
        @case ('code') {
          <path d="m8 9-3 3 3 3m8-6 3 3-3 3m-5 3 2-12" />
        }
        @case ('copy') {
          <rect x="8" y="8" width="11" height="11" rx="2" />
          <path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3" />
        }
        @case ('refresh') {
          <path d="M20 7v5h-5" />
          <path d="M18.5 16A8 8 0 1 1 20 12" />
        }
        @case ('close') {
          <path d="m6 6 12 12M18 6 6 18" />
        }
        @case ('file') {
          <path d="M6 2h8l4 4v16H6V2Z" />
          <path d="M14 2v5h5M9 12h6m-6 4h6" />
        }
        @case ('info') {
          <circle cx="12" cy="12" r="9" />
          <path d="M12 11v6m0-10h.01" />
        }
        @case ('plus') {
          <path d="M12 5v14M5 12h14" />
        }
        @case ('external') {
          <path d="M14 4h6v6M20 4l-9 9" />
          <path d="M18 13v7H4V6h7" />
        }
        @case ('eye') {
          <path d="M2 12s4-6 10-6 10 6 10 6-4 6-10 6S2 12 2 12Z" />
          <circle cx="12" cy="12" r="2.5" />
        }
        @case ('clock') {
          <circle cx="12" cy="12" r="9" />
          <path d="M12 7v5l3 2" />
        }
        @case ('trash') {
          <path d="M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v3" />
        }
        @case ('lock') {
          <rect x="4" y="11" width="16" height="10" rx="2" />
          <path d="M8 11V7a4 4 0 0 1 8 0v4" />
        }
      }
    </svg>
  `,
})
export class SvgIconComponent {
  readonly name = input.required<IconName>();
  readonly size = input<number>(18);
}
