import { TestBed } from '@angular/core/testing';
import { App } from './app';

describe('App', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({ imports: [App] }).compileComponents();
  });

  it('starts on the photo step with a drop target', async () => {
    const fixture = TestBed.createComponent(App);
    await fixture.whenStable();
    const el: HTMLElement = fixture.nativeElement;
    expect(el.querySelector('h1')?.textContent).toContain('Mesh out');
    expect(el.querySelector('.drop')).toBeTruthy();
    expect(el.querySelector('.flow li.on')?.textContent).toContain('Photo');
  });

  it('hides the Feedback button until feedback is set up', async () => {
    const fixture = TestBed.createComponent(App);
    await fixture.whenStable();
    expect(fixture.nativeElement.querySelector('.nav-link')).toBeNull();
  });
});
