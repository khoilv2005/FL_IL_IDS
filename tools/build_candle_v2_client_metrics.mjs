// Run a copy of this builder beside the runtime node_modules junction.
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const root = process.cwd();
const input = path.join(root, 'audit_denice/candle_v2_results15_20261022_v2');
const output = path.join(root, 'outputs/candle_v2_clients_results15_20261022');
const work = path.join(root, 'audit_denice/candle_v2_client_metrics_20261022');
await fs.mkdir(output, {recursive: true});
const sourceText = await fs.readFile(path.join(input, 'per_client.json'), 'utf8');
const completionText = await fs.readFile(path.join(input, 'completion.json'), 'utf8');
const clients = JSON.parse(sourceText).sort((a,b) => a.client_id-b.client_id);
const completed = JSON.parse(completionText);
if (!completed.completed || completed.evaluation_role !== 'full_TEST' || completed.inference_policy !== 'binary_cosine_registered_v1') {
  throw new Error('Expected completed full TEST of the registered binary-router repair');
}
const ids = clients.map(r => r.client_id);
if (clients.length !== 98 || new Set(ids).size !== 98) throw new Error('Invalid client membership');
const rows = clients.map(r => {
  const correct = Math.round(r.accuracy*r.samples);
  const routeCorrect = Math.round(r.route_accuracy*r.samples);
  if (Math.abs(correct/r.samples-r.accuracy) > 1e-12 || Math.abs(routeCorrect/r.samples-r.route_accuracy) > 1e-12) {
    throw new Error('Rates do not reconcile to integer counts');
  }
  return {...r, correct, routeCorrect};
});
const sum = key => rows.reduce((a,r) => a+r[key],0);
const n = sum('samples');
const accuracy = sum('correct')/n;
const routeAccuracy = sum('routeCorrect')/n;
if (n !== completed.test_rows || Math.abs(accuracy-completed.metrics.accuracy) > 1e-12 || Math.abs(routeAccuracy-completed.route_accuracy) > 1e-12) {
  throw new Error('Global/per-client accounting mismatch');
}
const workbook = Workbook.create();
const sheet = workbook.worksheets.add('Client metrics');
sheet.showGridLines = false;
sheet.tabColor = '#224966';
sheet.getRange('A1:J108').format.font = {name: 'Arial', size: 10, color: '#1F2937'};
sheet.getRange('A1:J108').format.verticalAlignment = 'center';
sheet.getRange('A2').values = [['CANDLE V2 — FULL TEST từng client']];
sheet.getRange('A2').format.font = {size: 15, bold: true, color: '#18364C'};
sheet.getRange('A2:J2').format.rowHeight = 28;
sheet.getRange('A3:J3').format.borders = {bottom: {style: 'thin', color: '#224966'}};
sheet.getRange('A4:I4').values = [['Clients','TEST rows','Accuracy','Route accuracy','Macro-F1 pooled','Coverage','Accuracy <10%','Mean client Macro-F1','Evaluation']];
sheet.getRange('A4:I4').format.font = {bold: true};
sheet.getRange('A4:I4').format.horizontalAlignment = 'center';
sheet.getRange('A4:I4').format.wrapText = true;
sheet.getRange('A4:I4').format.rowHeight = 35;
sheet.getRange('A5:I5').formulas = [[
  '=COUNTA(A11:A108)', '=SUM(F11:F108)', '=SUM(G11:G108)/B5', '=SUM(H11:H108)/B5',
  null, '=SUM(I11:I108)/B5', '=COUNTIFS(B11:B108,"<0.1")', '=AVERAGE(D11:D108)', null,
]];
sheet.getRange('E5').values = [[completed.metrics.macro_f1_34]];
sheet.getRange('I5').values = [['FULL TEST']];
sheet.getRange('A5:I5').format = {fill: '#E8F0F5', font: {bold: true}, rowHeight: 25};
sheet.getRange('A5:B5').setNumberFormat('#,##0');
sheet.getRange('C5:F5').setNumberFormat('0.00%');
sheet.getRange('G5').setNumberFormat('0');
sheet.getRange('H5').setNumberFormat('0.00%');
sheet.getRange('A7').values = [['T5/R19, seed 42. Mỗi TEST row được giao cho một client; không phải mỗi model chạy toàn bộ TEST.']];
sheet.getRange('A8').values = [['Nguồn: results (15), per_client.json và completion.json. Client 19 và 81 không có kết quả.']];
sheet.getRange('A7:J8').format.font = {size: 10, italic: true, color: '#526373'};
sheet.getRange('A7:J8').format.wrapText = false;
sheet.getRange('A10:J10').values = [['Client ID','Accuracy','Route accuracy','Client Macro-F1','Coverage','TEST rows','Đúng class','Đúng task','Rows có coverage','Accuracy <10%']];
sheet.getRange('A11:J108').values = rows.map(r => [r.client_id, null, null, r.macro_f1_34, null, r.samples, r.correct, r.routeCorrect, r.covered_rows, null]);
for (const [column, formula] of [['B','=G11/F11'],['C','=H11/F11'],['E','=I11/F11'],['J','=IF(B11<0.1,"Có","Không")']]) {
  sheet.getRange(`${column}11`).formulas = [[formula]];
  sheet.getRange(`${column}11:${column}108`).fillDown();
}
const table = sheet.tables.add('A10:J108', true, 'CandleV2ClientMetrics');
table.showFilterButton = true;
sheet.getRange('A10:J10').format = {fill: '#224966', font: {bold: true, color: '#FFFFFF'}, wrapText: true,
  horizontalAlignment: 'center', verticalAlignment: 'center', rowHeight: 34};
sheet.getRange('A11:J108').format.rowHeight = 21;
sheet.getRange('B11:E108').setNumberFormat('0.00%');
sheet.getRange('F11:I108').setNumberFormat('#,##0');
sheet.getRange('A11:A108').setNumberFormat('0');
sheet.getRange('A11:I108').format.horizontalAlignment = 'right';
sheet.getRange('J11:J108').format.horizontalAlignment = 'center';
sheet.getRange('B11:B108').conditionalFormats.add('cellIs', {operator: 'lessThan', formula: .1,
  format: {fill: '#FCE8E6', font: {bold: true, color: '#A52121'}}});
const widths = [12,14,17,18,14,15,15,15,18,17];
for (let c=0;c<10;c++) sheet.getRangeByIndexes(0,c,108,1).format.columnWidth = widths[c];
sheet.freezePanes.freezeRows(10);
sheet.freezePanes.freezeColumns(1);
workbook.recalculate();
const inspection = await workbook.inspect({kind: 'table', range: 'Client metrics!A4:I5', include: 'values,formulas', tableMaxRows: 3, tableMaxCols: 10, maxChars: 5000});
await fs.writeFile(path.join(work,'summary_inspection.jsonl'), inspection.ndjson);
const errors = await workbook.inspect({kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!', options: {useRegex:true,maxResults:30}, summary:'Final formula error scan'});
await fs.writeFile(path.join(work,'formula_error_scan.jsonl'), errors.ndjson);
const summary = sheet.getRange('A5:I5').values[0];
if (summary[0] !== 98 || summary[1] !== n || Math.abs(summary[2]-accuracy)>1e-12 || Math.abs(summary[3]-routeAccuracy)>1e-12 || summary[6] !== 3) {
  throw new Error('Workbook summary does not reconcile');
}
for (let i=0;i<rows.length;i++) {
  const cells = sheet.getRange(`A${i+11}:J${i+11}`).values[0];
  if (cells[0]!==rows[i].client_id || Math.abs(cells[1]-rows[i].accuracy)>1e-12 || Math.abs(cells[2]-rows[i].route_accuracy)>1e-12) throw new Error('Export table row mismatch');
}
const preview = await workbook.render({sheetName:'Client metrics',range:'A1:J20',scale:1.5,format:'png'});
await fs.writeFile(path.join(work,'client_metrics_preview.png'),new Uint8Array(await preview.arrayBuffer()));
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
const deliverable = path.join(output,'CANDLE_V2_FULL_TEST_98_CLIENTS.xlsx');
await xlsx.save(deliverable);
const pct = v => (100*v).toFixed(2)+'%';
const report = [
  '# CANDLE V2 — FULL TEST accuracy và route accuracy của từng client', '',
  'Nguồn: results (15), T5/R19, `binary_cosine_registered_v1`, seed 42.',
  `Tổng ${n.toLocaleString('en-US')} TEST rows, 98 local predictors; mỗi row giao đúng một client.`,
  `Accuracy pooled: **${pct(accuracy)}**; route accuracy pooled: **${pct(routeAccuracy)}**.`,
  'Không có bản ghi client 19 hoặc 81; không điền số 0 thay cho kết quả thiếu.', '',
  '| Client | Accuracy | Route accuracy | Macro-F1 | Coverage |', '|---|---:|---:|---:|---:|',
  ...rows.map(r=>`| ${r.client_id} | ${pct(r.accuracy)} | ${pct(r.route_accuracy)} | ${pct(r.macro_f1_34)} | ${pct(r.covered_rows/r.samples)} |`), '',
  'Macro-F1 trong từng dòng là metric của client. Pooled Macro-F1 toàn hệ thống là 18,48%; mean client Macro-F1 là 14,06%.',
  'Coverage là tỷ lệ TEST rows có true class nằm trong saved availability của receiver. Không phải route accuracy.',
  'Ba client accuracy dưới 10%: 38, 32, 47. Đây là số FULL TEST đã lưu, không phải development hoặc oracle.', '',
];
await fs.writeFile(path.join(root,'docs/CANDLE_V2_ALL_CLIENT_METRICS_20261022_VI.md'),report.join('\n'));
const receipt = {
  completed:true, evaluation_role:'full_TEST', inference_policy:completed.inference_policy,
  sources:Object.fromEntries([['per_client.json',sourceText],['completion.json',completionText]].map(([k,v])=>[k,crypto.createHash('sha256').update(v).digest('hex')])),
  client_count:rows.length,missing_client_ids:[19,81],rows:n,accuracy,route_accuracy:routeAccuracy,
  pooled_macro_f1:completed.metrics.macro_f1_34,mean_client_macro_f1:completed.mean_client_macro_f1,
  low_accuracy_clients:rows.filter(r=>r.accuracy<.1).map(r=>r.client_id),
  no_training:true,no_new_inference:true,
  clients:rows.map(r=>({client_id:r.client_id,accuracy:r.accuracy,route_accuracy:r.route_accuracy,macro_f1:r.macro_f1_34,coverage:r.covered_rows/r.samples,samples:r.samples,correct:r.correct,route_correct:r.routeCorrect,covered_rows:r.covered_rows})),
};
await fs.writeFile(path.join(root,'artifacts/candle_v2_all_client_metrics_20261022.json'),JSON.stringify(receipt,null,2)+'\n');
console.log(JSON.stringify({deliverable,clients:rows.length,rows:n,accuracy,routeAccuracy,lowClients:receipt.low_accuracy_clients},null,2));
