import { Pipe, PipeTransform, inject } from '@angular/core';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import DOMPurify from 'dompurify';
import { marked } from 'marked';

let purifierInitialized = false;

function ensurePurifierHooks(): void {
  if (purifierInitialized) {
    return;
  }
  purifierInitialized = true;

  DOMPurify.addHook('afterSanitizeAttributes', (node) => {
    if (node.tagName === 'A') {
      const href = node.getAttribute('href') ?? '';
      if (!/^https?:\/\//i.test(href) && !href.startsWith('#')) {
        node.removeAttribute('href');
      } else {
        node.setAttribute('target', '_blank');
        node.setAttribute('rel', 'noopener noreferrer');
      }
    }
  });
}

export function sanitizeMarkdownToHtml(markdown: string): string {
  ensurePurifierHooks();
  const rawHtml = marked.parse(markdown || '', {
    gfm: true,
    breaks: true,
    async: false,
  }) as string;

  return DOMPurify.sanitize(rawHtml, {
    ALLOWED_TAGS: [
      'p',
      'br',
      'strong',
      'b',
      'em',
      'i',
      'code',
      'pre',
      'ul',
      'ol',
      'li',
      'blockquote',
      'h1',
      'h2',
      'h3',
      'h4',
      'h5',
      'h6',
      'table',
      'thead',
      'tbody',
      'tr',
      'th',
      'td',
      'hr',
      'a',
    ],
    ALLOWED_ATTR: ['href', 'title', 'target', 'rel'],
    FORBID_TAGS: ['script', 'style', 'iframe', 'object', 'embed', 'form', 'input', 'svg', 'math'],
    ALLOW_DATA_ATTR: false,
  });
}

@Pipe({
  name: 'safeMarkdown',
  standalone: true,
  pure: true,
})
export class SafeMarkdownPipe implements PipeTransform {
  private readonly sanitizer = inject(DomSanitizer);

  transform(markdown: string | null | undefined): SafeHtml {
    if (!markdown) {
      return '';
    }
    const cleanHtml = sanitizeMarkdownToHtml(markdown);
    return this.sanitizer.bypassSecurityTrustHtml(cleanHtml);
  }
}
