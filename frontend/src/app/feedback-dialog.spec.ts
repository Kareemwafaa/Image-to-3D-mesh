import { TestBed } from '@angular/core/testing';
import { FeedbackDialog } from './feedback-dialog';
import { FeedbackService } from './feedback.service';

describe('FeedbackDialog', () => {
  const send = vi.fn().mockResolvedValue(undefined);

  beforeEach(async () => {
    send.mockClear();
    // jsdom has no <dialog> methods.
    HTMLDialogElement.prototype.showModal ??= function (this: HTMLDialogElement) {
      this.open = true;
    };
    HTMLDialogElement.prototype.close ??= function (this: HTMLDialogElement) {
      this.open = false;
    };
    await TestBed.configureTestingModule({
      imports: [FeedbackDialog],
      providers: [{ provide: FeedbackService, useValue: { send, load: () => Promise.resolve() } }],
    }).compileComponents();
  });

  it('offers the attachment only when there is a photo, and leaves it off', async () => {
    const fixture = TestBed.createComponent(FeedbackDialog);
    await fixture.whenStable();
    const el: HTMLElement = fixture.nativeElement;
    expect(el.querySelector('input[name=attach]')).toBeNull();

    fixture.componentRef.setInput('photo', new File(['x'], 'a.png', { type: 'image/png' }));
    await fixture.whenStable();
    const box = el.querySelector<HTMLInputElement>('input[name=attach]');
    expect(box).toBeTruthy();
    expect(box!.checked).toBe(false);
  });

  it('sends the text without files unless the box is ticked', async () => {
    const fixture = TestBed.createComponent(FeedbackDialog);
    fixture.componentRef.setInput('photo', new File(['x'], 'a.png', { type: 'image/png' }));
    await fixture.whenStable();
    const el: HTMLElement = fixture.nativeElement;
    const submit = el.querySelector<HTMLButtonElement>('button[type=submit]')!;
    expect(submit.disabled).toBe(true);

    el.querySelectorAll<HTMLButtonElement>('.stars button')[3].click();
    await fixture.whenStable();
    expect(submit.disabled).toBe(false);
    el.querySelector('form')!.dispatchEvent(new Event('submit'));
    await fixture.whenStable();

    expect(send).toHaveBeenCalledOnce();
    expect(send.mock.calls[0][0]).toMatchObject({ rating: 4, photo: null, model: null });
    expect(el.textContent).toContain('Thanks');
  });
});
