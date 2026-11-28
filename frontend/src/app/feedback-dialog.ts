import { Component, ElementRef, computed, inject, input, signal, viewChild } from '@angular/core';
import { EMAIL_MAX, FeedbackService, MESSAGE_MAX } from './feedback.service';

/** The "Feedback" popup: a rating, a note, and an opt-in to attach the photo and model. */
@Component({
  selector: 'app-feedback-dialog',
  template: `
    <dialog
      #dialog
      class="server-dialog feedback-dialog"
      aria-labelledby="feedback-title"
      (click)="$event.target === dialog ? close() : null"
      (close)="onClosed()"
    >
      @if (state() === 'sent') {
        <div class="feedback-sent">
          <span class="sent-mark" aria-hidden="true">
            <svg viewBox="0 0 24 24">
              <path
                d="m6 12.5 4 4L18 8"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
            </svg>
          </span>
          <h2 id="feedback-title">Thanks, that helps.</h2>
          <p class="dialog-note">
            Your feedback was sent{{ sentFiles() ? ' with the photo and model' : '' }}.
          </p>
          <div class="dialog-actions">
            <span class="spacer"></span
            ><button type="button" class="btn primary" (click)="close()">Done</button>
          </div>
        </div>
      } @else {
        <form (submit)="submit($event)">
          <h2 id="feedback-title">Send feedback</h2>
          <p class="dialog-note">How did it go? Bugs, odd shapes and ideas are all welcome.</p>

          <fieldset class="rating">
            <legend class="option-name">
              {{ modelUrl() ? 'How good is this model?' : 'How is Facet so far?' }}
            </legend>
            <div class="stars" role="radiogroup" aria-label="Rating">
              @for (n of [1, 2, 3, 4, 5]; track n) {
                <button
                  type="button"
                  role="radio"
                  [attr.aria-checked]="rating() === n"
                  [attr.aria-label]="n + (n === 1 ? ' star' : ' stars')"
                  [class.on]="(rating() ?? 0) >= n"
                  (click)="rating.set(rating() === n ? null : n)"
                >
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path
                      d="m12 3.5 2.6 5.3 5.9.9-4.3 4.1 1 5.8L12 16.9l-5.2 2.7 1-5.8-4.3-4.1 5.9-.9z"
                      stroke-linejoin="round"
                    />
                  </svg>
                </button>
              }
              <span class="rating-label mono">{{ ratingLabel() }}</span>
            </div>
          </fieldset>

          <label class="field">
            <span class="option-name">What should we know?</span>
            <textarea
              name="message"
              rows="4"
              [attr.maxlength]="messageMax"
              placeholder="e.g. the handle came out thin, or the back is flat"
              [value]="message()"
              (input)="message.set($any($event.target).value)"
            ></textarea>
          </label>

          <label class="field">
            <span class="option-name"
              >Email <span class="optional">optional, only if you'd like a reply</span></span
            >
            <input
              type="email"
              name="email"
              autocomplete="email"
              [attr.maxlength]="emailMax"
              placeholder="you@example.com"
              [value]="email()"
              (input)="email.set($any($event.target).value)"
            />
          </label>

          <!-- Left empty by people; bots that fill every field are dropped. -->
          <input
            class="visually-hidden"
            type="text"
            name="website"
            tabindex="-1"
            autocomplete="off"
            aria-hidden="true"
            (input)="trap.set($any($event.target).value)"
          />

          @if (photo()) {
            <label class="option attach">
              <span class="option-text">
                <span class="option-name"
                  >Include my photo{{ modelUrl() ? ' and the 3D model' : '' }}</span
                >
                <span class="option-note">
                  They are stored privately and only used to understand your feedback and improve
                  results. Leave this off to send just your message.
                </span>
              </span>
              <span class="switch">
                <input
                  type="checkbox"
                  name="attach"
                  [checked]="attach()"
                  (change)="attach.set($any($event.target).checked)"
                />
                <span class="track"></span>
              </span>
            </label>
          }

          @if (error(); as e) {
            <p class="dialog-status" data-state="fail" role="alert">{{ e }}</p>
          }

          <div class="dialog-actions">
            <span class="spacer"></span>
            <button type="button" class="btn quiet" (click)="close()">Cancel</button>
            <button
              type="submit"
              class="btn primary"
              [disabled]="state() === 'sending' || !canSend()"
            >
              {{ state() === 'sending' ? 'Sending…' : 'Send feedback' }}
            </button>
          </div>
          <p class="dialog-foot">Only the Facet maintainer can read what you send.</p>
        </form>
      }
    </dialog>
  `,
})
export class FeedbackDialog {
  private readonly feedback = inject(FeedbackService);
  private readonly dialog = viewChild.required<ElementRef<HTMLDialogElement>>('dialog');

  /** The photo in use, if any; offered as an attachment. */
  readonly photo = input<File | null>(null);
  /** Object URL of the generated .glb, if a model is showing. */
  readonly modelUrl = input<string | null>(null);
  /** Run settings sent with every message. */
  readonly context = input<Record<string, unknown>>({});

  protected readonly messageMax = MESSAGE_MAX;
  protected readonly emailMax = EMAIL_MAX;
  protected readonly rating = signal<number | null>(null);
  protected readonly message = signal('');
  protected readonly email = signal('');
  protected readonly attach = signal(false);
  protected readonly trap = signal('');
  protected readonly state = signal<'idle' | 'sending' | 'sent'>('idle');
  protected readonly error = signal<string | null>(null);
  protected readonly sentFiles = signal(false);
  protected readonly canSend = computed(
    () => this.rating() !== null || this.message().trim().length > 0,
  );
  protected readonly ratingLabel = computed(
    () => ['', 'Poor', 'Not great', 'Okay', 'Good', 'Great'][this.rating() ?? 0] || 'Tap a star',
  );

  open() {
    if (this.state() === 'sent') this.resetForm();
    this.error.set(null);
    this.dialog().nativeElement.showModal();
  }

  close() {
    this.dialog().nativeElement.close();
  }

  protected onClosed() {
    if (this.state() === 'sent') this.resetForm();
  }

  protected async submit(e: Event) {
    e.preventDefault();
    if (!this.canSend() || this.state() === 'sending') return;
    this.state.set('sending');
    this.error.set(null);
    const withFiles = this.attach() && !!this.photo();
    try {
      if (!this.trap()) {
        await this.feedback.send({
          rating: this.rating(),
          message: this.message(),
          email: this.email(),
          photo: withFiles ? this.photo() : null,
          model: withFiles ? await this.modelBlob() : null,
          context: this.context(),
        });
      }
      this.sentFiles.set(withFiles);
      this.state.set('sent');
    } catch (err) {
      this.error.set(err instanceof Error ? err.message : String(err));
      this.state.set('idle');
    }
  }

  private async modelBlob(): Promise<Blob | null> {
    const url = this.modelUrl();
    if (!url) return null;
    const blob = await (await fetch(url)).blob();
    return new Blob([blob], { type: 'model/gltf-binary' });
  }

  private resetForm() {
    this.rating.set(null);
    this.message.set('');
    this.email.set('');
    this.attach.set(false);
    this.state.set('idle');
    this.sentFiles.set(false);
  }
}
