import { ChangeDetectionStrategy, Component, input } from '@angular/core';
import { IconName, SvgIconComponent } from './svg-icon.component';

@Component({
  selector: 'sb-kpi-card',
  standalone: true,
  imports: [SvgIconComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="kpi">
      <div class="kpi-top">
        <span>{{ label() }}</span>
        <sb-svg-icon [name]="icon()" [size]="16" />
      </div>
      <strong>{{ value() }}</strong>
      <small>{{ note() }}</small>
    </div>
  `,
})
export class KpiCardComponent {
  readonly label = input.required<string>();
  readonly value = input.required<string>();
  readonly note = input.required<string>();
  readonly icon = input.required<IconName>();
}
