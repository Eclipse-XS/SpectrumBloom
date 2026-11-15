import {
  formatBytes,
  formatDurationMs,
  formatInteger,
  formatNumber,
  getFileExtensionBadge,
  sanitizeCsvCell,
  slugifyFilename,
} from './format.utils';

describe('format.utils', () => {
  it('neutralizes CSV spreadsheet formula prefixes', () => {
    expect(sanitizeCsvCell('=SUM(A1:A2)')).toBe("'=SUM(A1:A2)");
    expect(sanitizeCsvCell('+cmd|"/C calc"!A0')).toBe('"\'+cmd|""/C calc""!A0"');
    expect(sanitizeCsvCell('-2+3')).toBe("'-2+3");
    expect(sanitizeCsvCell('@HYPERLINK("http://evil")')).toBe('"\'@HYPERLINK(""http://evil"")"');
    expect(sanitizeCsvCell(' \t=1+1')).toBe("' \t=1+1");
    expect(sanitizeCsvCell('normal, value')).toBe('"normal, value"');
    expect(sanitizeCsvCell(null)).toBe('');
  });

  it('formats integers, decimals, bytes, durations, and file badges safely', () => {
    expect(formatInteger(1234)).toBe('1,234');
    expect(formatInteger(null)).toBe('—');
    expect(formatNumber(3.14159, 2)).toBe('3.14');
    expect(formatBytes(1536)).toBe('1.5 KB');
    expect(formatDurationMs(420)).toBe('420 ms');
    expect(formatDurationMs(2450)).toBe('2.45 s');
    expect(getFileExtensionBadge('sales_2025.xlsx')).toBe('XLSX');
    expect(slugifyFilename('Sample E-Commerce Data.csv')).toBe('sample_e_commerce_data_csv');
  });
});
