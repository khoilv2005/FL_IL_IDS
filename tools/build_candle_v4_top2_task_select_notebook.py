"""Separate experimental eval notebook; single-task Top2 reranking, no union."""
import argparse
import json
from pathlib import Path
from build_candle_v4_round19_topk_notebook import SOURCE


def build(ref):
    source=SOURCE.replace('__REF__',ref)
    source=source.replace('eval_candle_v4_round19_top1_top2','eval_candle_v4_round19_top2_task_select')
    source=source.replace('candle_v4_six_task_topk','candle_v4_six_task_select')
    source=source.replace("'/tools/eval_candle_v4_task_round19_topk.py','/tools/candle_v4_historical_repair.py'",
        "'/tools/eval_candle_v4_task_round19_topk.py','/tools/candle_v4_historical_repair.py','/tools/candle_v4_top2_task_select.py'")
    marker="'--cache',str(cache/'continuations'),'--output',OUTPUT_DIR]"
    if marker not in source:raise ValueError('Builder command changed')
    source=source.replace(marker,"'--cache',str(cache/'continuations'),'--output',OUTPUT_DIR,'--top2-policy','mass']")
    compile(source,'notebook','exec')
    return source


def main():
    p=argparse.ArgumentParser();p.add_argument('--ref',required=True);a=p.parse_args()
    if len(a.ref)!=40 or any(c not in '0123456789abcdef' for c in a.ref):raise ValueError('Full public SHA required')
    source=build(a.ref)
    text='''# CANDLE V4 — Top2 chọn một task, không mở union

Chỉ eval checkpoint Round19 sau consolidation của T0–T5; không train/refit.
GPU + Internet, gắn100-clients, Run All. results19/21 đã có link Drive tự tải.

**Quy tắc Top2 mới (experimental):**

1. Router gốc chọn hai task ứng viên có memory từ input.
2. Cộng các xác suất class LR đã lưu theo task, chọn task có tổng lớn hơn trong hai ứng viên.
3. Ties hoặc không có ứng viên thứ hai: giữ Top1.
4. Chỉ chạy adapter, native mask và experts của task được chọn, threshold expert0,5.
5. Không dùng nhãn thật, không mở đồng thời class/experts của hai task.

Giữ Top1 làm đối chứng; T5 Top1 phải khớp results21 từng client. Cumulative TEST và
fixed34-label Macro-F1 giữ như notebook cũ; có thêm seen-class Macro-F1, selected-task
route accuracy, candidate Top2 recall, rescue/break. Không nhầm recall với accuracy.

**Giới hạn đã biết trước khi chạy:** trên panel development204.800 mẫu, Top1=48,0718%,
mass-rerank=48,0654%; Macro-F1 27,5621% ->27,6294%. Đường mới đã tránh sụt lớn của
union nhưng chưa cải thiện Accuracy so với Top1. Không kỳ vọng oracle61,30%.
Đây là readout experimental, không phải bản Top2 đã chứng minh tốt hơn hoặc confirmation độc lập.
Không cần chạy training lại để dùng notebook này. Notebook union cũ giữ nguyên để tái lập results22.

Output: `/kaggle/working/eval_candle_v4_round19_top2_task_select`.
Full eval6 checkpoints sẽ tốn thời gian; local regressions không thay thế full result.
Save Version để giữ summary.csv, task reports và completion.
'''
    n=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python')),
        cells=[dict(cell_type='markdown',metadata={},source=text.splitlines(True)),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=source.splitlines(True))])
    root=Path(__file__).resolve().parents[1];stem='eval_candle_v4_top2_task_select_kaggle'
    (root/(stem+'.py')).write_text(source,encoding='utf-8')
    (root/(stem+'.ipynb')).write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print('Created',stem+'.ipynb')


if __name__=='__main__':main()
