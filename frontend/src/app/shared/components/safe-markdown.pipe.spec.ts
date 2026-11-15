import { sanitizeMarkdownToHtml, SafeMarkdownPipe } from './safe-markdown.pipe';
import { TestBed } from '@angular/core/testing';

describe('SafeMarkdownPipe', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [SafeMarkdownPipe],
    });
  });

  it('strips dangerous script, iframe, and javascript: links', () => {
    const dirty = [
      '# Analysis',
      '<script>window.__xss = true;</script>',
      '<iframe src="https://evil.example"></iframe>',
      '[Click me](javascript:alert(1))',
      '[Safe doc](https://example.com/docs)',
      '<img src="x" onerror="alert(1)">',
    ].join('\n\n');

    const cleaned = sanitizeMarkdownToHtml(dirty);
    expect(cleaned).not.toContain('<script');
    expect(cleaned).not.toContain('<iframe');
    expect(cleaned).not.toContain('javascript:');
    expect(cleaned).not.toContain('onerror');
    expect(cleaned).toContain('<h1');
    expect(cleaned).toContain('href="https://example.com/docs"');
    expect(cleaned).toContain('rel="noopener noreferrer"');
  });

  it('renders GFM tables and returns empty string for nullish input', () => {
    const pipe = TestBed.inject(SafeMarkdownPipe);
    expect(pipe.transform(null)).toBe('');
    expect(pipe.transform('')).toBe('');

    const tableHtml = sanitizeMarkdownToHtml('| col | val |\n| --- | --- |\n| rev | 42 |');
    expect(tableHtml).toContain('<table>');
    expect(tableHtml).toContain('42');
  });
});
