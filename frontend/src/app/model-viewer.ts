import { Component, CUSTOM_ELEMENTS_SCHEMA, input, OnInit, signal } from '@angular/core';

// Wraps Google's <model-viewer> web component. The library is imported
// lazily so it stays out of the initial bundle.
@Component({
  selector: 'app-model-viewer',
  schemas: [CUSTOM_ELEMENTS_SCHEMA],
  template: `
    @if (ready()) {
      <model-viewer
        class="viewer"
        [attr.src]="src()"
        [attr.alt]="alt()"
        camera-controls
        auto-rotate
        shadow-intensity="1"
        touch-action="pan-y"
      ></model-viewer>
    } @else {
      <div class="viewer placeholder">Loading viewer…</div>
    }
  `,
})
export class ModelViewer implements OnInit {
  readonly src = input.required<string>();
  readonly alt = input('3D model');
  protected readonly ready = signal(false);

  ngOnInit() {
    import('@google/model-viewer').then(() => this.ready.set(true));
  }
}
