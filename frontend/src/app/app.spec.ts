import { TestBed } from '@angular/core/testing';
import { App } from './app';

describe('App', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({ imports: [App] }).compileComponents();
  });

  it('renders the upload step with Generate disabled', async () => {
    const fixture = TestBed.createComponent(App);
    await fixture.whenStable();
    const el: HTMLElement = fixture.nativeElement;
    expect(el.querySelector('.brand-name')?.textContent).toContain('Image to 3D');
    const btn = el.querySelector<HTMLButtonElement>('button.btn.primary');
    expect(btn?.disabled).toBe(true);
  });
});
