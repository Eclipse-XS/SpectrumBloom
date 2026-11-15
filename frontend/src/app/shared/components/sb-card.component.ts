import { ChangeDetectionStrategy, Component, input } from '@angular/core';

@Component({
  selector: 'sb-card',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <section [class]="'card ' + cardClass()">
      @if (title()) {
        <header class="card-head">
          <div>
            <h2>{{ title() }}</h2>
            @if (subtitle()) {
              <p>{{ subtitle() }}</p>
            }
          </div>
          <div class="card-actions">
            <ng-content select="[card-actions]" />
          </div>
        </header>
      }
      <div class="card-body">
        <ng-content />
      </div>
    </section>
  `,
})
export class SbCardComponent {
  readonly title = input<string>('');
  readonly subtitle = input<string>('');
  readonly cardClass = input<string>('');
}
