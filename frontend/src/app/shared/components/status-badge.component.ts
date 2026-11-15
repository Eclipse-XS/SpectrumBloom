import { ChangeDetectionStrategy, Component, input } from '@angular/core';
import { StatusKind } from '../../core/models/workspace.models';

@Component({
  selector: 'sb-status-badge',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <span [class]="'status ' + kind()">
      <i aria-hidden="true"></i>
      <ng-content />
    </span>
  `,
})
export class StatusBadgeComponent {
  readonly kind = input<StatusKind>('success');
}
