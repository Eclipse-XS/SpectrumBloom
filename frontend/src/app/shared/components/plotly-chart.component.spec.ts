import { Component, signal } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { PlotlyChartComponent } from './plotly-chart.component';

@Component({
  standalone: true,
  imports: [PlotlyChartComponent],
  template: `<sb-plotly-chart [figureJson]="figJson()" [title]="'Revenue Distribution'" [height]="260" />`,
})
class TestHostComponent {
  readonly figJson = signal(
    JSON.stringify({
      data: [{ type: 'bar', x: ['North', 'South'], y: [120, 85] }],
      layout: { title: { text: 'Regional Totals' } },
    }),
  );
}

describe('PlotlyChartComponent', () => {
  it('renders a live Plotly chart container for valid figure JSON', async () => {
    await TestBed.configureTestingModule({
      imports: [TestHostComponent],
    }).compileComponents();

    const fixture = TestBed.createComponent(TestHostComponent);
    fixture.detectChanges();

    // Wait for async dynamic import('plotly.js-dist-min') and Plotly.react
    for (let i = 0; i < 40; i++) {
      await new Promise((resolve) => setTimeout(resolve, 100));
      fixture.detectChanges();
      const plotEl = fixture.nativeElement.querySelector('.js-plotly-plot');
      if (plotEl) {
        expect(plotEl).toBeTruthy();
        fixture.destroy();
        return;
      }
    }
    fail('Expected .js-plotly-plot to render within timeout');
  });

  it('shows a graceful error state when figure JSON is invalid', async () => {
    await TestBed.configureTestingModule({
      imports: [TestHostComponent],
    }).compileComponents();

    const fixture = TestBed.createComponent(TestHostComponent);
    fixture.componentInstance.figJson.set('{invalid-json');
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();

    const alert = fixture.nativeElement.querySelector('[role="alert"]');
    expect(alert?.textContent).toContain('Invalid Plotly figure JSON payload');
  });
});
