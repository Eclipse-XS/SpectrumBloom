import { ChangeDetectionStrategy, Component, input } from '@angular/core';

@Component({
  selector: 'sb-page-header',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <header class="page-title">
      <div>
        @if (eyebrow()) {
          <div class="eyebrow">{{ eyebrow() }}</div>
        }
        <h1>{{ title() }}</h1>
        <p>{{ description() }}</p>
      </div>
      <div class="page-actions">
        <ng-content />
      </div>
    </header>
  `,
})
export class PageHeaderComponent {
  readonly eyebrow = input<string>('');
  readonly title = input.required<string>();
  readonly description = input.required<string>();
}
