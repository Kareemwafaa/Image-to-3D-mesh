import { Component, CUSTOM_ELEMENTS_SCHEMA, input, OnInit, signal } from '@angular/core';

// Wraps Google's <model-viewer> web component. The library is imported
// lazily so it stays out of the initial bundle.
@Component({
  selector: 'app-model-viewer',
  schemas: [CUSTOM_ELEMENTS_SCHEMA],
  host: { class: 'model-host' },
  template: `
    @if (ready()) {
      <model-viewer
        [attr.src]="src()"
        [attr.alt]="alt()"
        camera-controls
        [attr.auto-rotate]="autoRotate() ? '' : null"
        auto-rotate-delay="1200"
        rotation-per-second="18deg"
        interaction-prompt="none"
        environment-image="neutral"
        shadow-intensity="0.9"
        shadow-softness="0.8"
        exposure="1.05"
        touch-action="pan-y"
      >
        <div slot="progress-bar"></div>
      </model-viewer>
    }
  `,
})
export class ModelViewer implements OnInit {
  readonly src = input.required<string>();
  readonly alt = input('3D model');
  readonly autoRotate = input(true);
  protected readonly ready = signal(false);

  ngOnInit() {
    import('@google/model-viewer').then(() => this.ready.set(true));
  }
}
