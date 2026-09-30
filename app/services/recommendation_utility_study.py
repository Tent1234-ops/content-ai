"""Pure helpers for the blinded Phase 7 recommendation utility study."""
from __future__ import annotations

import html
import json
from pathlib import Path

from app.services.analysis_evaluation import (
    CRITICAL_FLAGS,
    EVIDENCE_SCORE_FIELD,
    REVIEW_SCHEMA_VERSION,
    UTILITY_SCORE_FIELDS,
    allocation_balance,
    digest,
)


def stable_context_lock(context: dict) -> dict:
    """Remove observation time while retaining every input that can change an output."""
    keys = (
        "settings", "artifact_path", "artifact_sha256", "dataset_manifest_sha256",
        "reference_sha256", "reference_count", "statistics_cutoff",
        "method_versions", "packages", "python_version", "asr_language", "code_sha256",
    )
    return {key: context.get(key) for key in keys}


def study_readiness(*, manifest_audit: dict, reviewer_ids: list[str]) -> dict:
    missing = []
    for label, count in manifest_audit.get("eligible_counts", {}).items():
        if count < 3:
            missing.append({"requirement": f"fresh_{label}_clips", "required": 3, "available": count})
    if not manifest_audit.get("diversity_complete"):
        missing.append({"requirement": "pilot_scenario_and_self_recorded_coverage",
                        "required": 1, "available": 0})
    if len(set(reviewer_ids)) < 3:
        missing.append({"requirement": "real_reviewers", "required": 3,
                        "available": len(set(reviewer_ids))})
    return {
        "state": "ready_for_prediction" if not missing else "awaiting_fresh_clips_or_reviewers",
        "manifest_complete": bool(manifest_audit.get("pilot_manifest_complete")),
        "reviewer_plan_complete": len(set(reviewer_ids)) >= 3,
        "missing": missing,
    }


def build_blind_packets(*, protocol: dict, allocations: list[dict], case_outputs: dict) -> tuple[dict, dict]:
    """Return public packets and a private key; packets never reveal A/B/C."""
    packets: dict[str, dict] = {}
    private_rows = []
    for row in allocations:
        case_id = row["case_id"]
        reviewer_id = row["reviewer_id"]
        variant_id = row["variant_id"]
        case = case_outputs[case_id]
        variant = case["variants"][variant_id]
        packet = packets.setdefault(reviewer_id, {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "study_id": protocol["study_id"],
            "protocol_sha256": protocol["protocol_sha256"],
            "reviewer_id": reviewer_id,
            "instructions": {
                "blind": "รหัสชุดนำเสนอไม่บอกว่าเป็น A, B หรือ C",
                "sequence": "ดูคลิปและตอบคำถามก่อน จากนั้นประเมินคำแนะนำทีละชุดตามลำดับ",
                "scale": "1=ต่ำมาก, 5=สูงมาก; คะแนนหลักฐานใช้เฉพาะชุดที่มีหลักฐาน",
                "claims": "ประเมินประโยชน์ของคำแนะนำ ไม่ใช่ทำนายยอด Engagement",
            },
            "cases": {},
        })
        public_case = packet["cases"].setdefault(case_id, {
            "case_id": case_id,
            "video_uri": case.get("video_uri"),
            "gold_label_hidden": True,
            "analysis_status": case.get("analysis_status"),
            "presentations": [],
        })
        unit_id = digest([protocol["study_id"], case_id, reviewer_id,
                          row["presentation_id"], "rating-unit"])[:24]
        presentation = {
            "presentation_id": row["presentation_id"],
            "position": row["position"],
            "unit_id": unit_id,
            "output_sha256": variant["output_sha256"],
            "item_count": variant["item_count"],
            "recommendation_status": variant.get("recommendation_status"),
            "items": variant["items"],
            "evidence_score_applicable": variant_id == "C" and variant["item_count"] > 0,
        }
        public_case["presentations"].append(presentation)
        private_rows.append({**row, "output_sha256": variant["output_sha256"],
                             "unit_id": unit_id})
    for packet in packets.values():
        packet["cases"] = [packet["cases"][key] for key in sorted(packet["cases"])]
        for case in packet["cases"]:
            case["presentations"].sort(key=lambda item: item["position"])
        packet["packet_sha256"] = digest(packet)
    private = {
        "study_id": protocol["study_id"],
        "protocol_sha256": protocol["protocol_sha256"],
        "allocation_balance": allocation_balance(allocations),
        "allocations": private_rows,
        "private_mapping_sha256": digest(private_rows),
    }
    return packets, private


def _json_for_script(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def render_review_html(packet: dict) -> str:
    """Create a standalone, sequential review form that downloads signed response JSON."""
    payload = _json_for_script(packet)
    utility_fields = json.dumps(list(UTILITY_SCORE_FIELDS))
    flags = json.dumps(list(CRITICAL_FLAGS))
    return f"""<!doctype html>
<html lang="th"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>แบบประเมินคำแนะนำ</title><style>
body{{font-family:Arial,sans-serif;margin:0;background:#f5f7fa;color:#152033}}main{{max-width:820px;margin:auto;padding:24px}}
.panel{{background:white;border:1px solid #d8dee8;border-radius:8px;padding:20px;margin:16px 0}}button{{padding:10px 16px}}
video{{width:100%;max-height:420px;background:#111}}label{{display:block;margin:12px 0}}select,textarea{{width:100%;padding:8px}}
.item{{border-top:1px solid #ddd;padding-top:12px;margin-top:12px}}.muted{{color:#657086}}.hidden{{display:none}}
</style></head><body><main><h1>แบบประเมินคำแนะนำสำหรับคลิป</h1>
<p>ดูคลิปก่อนตอบ แล้วระบบจะแสดงคำแนะนำทีละชุดโดยไม่เปิดเผยรูปแบบ A/B/C</p>
<section id="root" class="panel"></section></main><script>
const packet={payload}; const utilityFields={utility_fields}; const criticalFlags={flags};
const response={{schema_version:'{REVIEW_SCHEMA_VERSION}',study_id:packet.study_id,protocol_sha256:packet.protocol_sha256,
 reviewer_id:packet.reviewer_id,consent:false,pre_reviews:[],ratings:[]}};
let caseIndex=0, stage='consent', presentationIndex=0; const root=document.getElementById('root');
const esc=s=>String(s??'').replace(/[&<>\"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}}[c]));
function score(name,label){{return `<label>${{label}}<select id="${{name}}"><option value="">เลือก 1–5</option>${{[1,2,3,4,5].map(n=>`<option>${{n}}</option>`).join('')}}</select></label>`}}
function renderItems(p){{if(!p.items.length)return '<p><strong>ระบบงดคำแนะนำในชุดนี้</strong></p>';
 return p.items.map(x=>`<div class="item"><h3>${{esc(x.title)}}</h3>${{x.action?`<p>${{esc(x.action.finding)}}</p><p><strong>เสนอ:</strong> ${{esc(x.action.proposal)}}</p><p><strong>วิธีทำ:</strong> ${{esc((x.action.steps||[]).join(' / '))}}</p><p><strong>ตัวอย่าง:</strong> ${{esc(x.action.example)}}</p>`:''}}${{x.evidence?`<p><strong>เหตุผล/หลักฐาน:</strong> ${{esc(x.evidence.reason)}}</p><p class="muted">ตัวอย่างสนับสนุน ${{esc(x.evidence.support_count)}} คลิป จาก ${{esc(x.evidence.channel_count)}} ช่อง; ${{esc(x.evidence.limitation)}}</p>`:''}}</div>`).join('')}}
function render(){{const c=packet.cases[caseIndex]; if(stage==='consent'){{root.innerHTML=`<h2>ยินยอมเข้าร่วม</h2><p>รหัสผู้ประเมิน: ${{esc(packet.reviewer_id)}}</p><label><input id="consent" type="checkbox"> ยินยอมให้ใช้คะแนนแบบไม่เปิดเผยตัวตน</label><button onclick="start()">เริ่ม</button>`;return}}
 if(!c){{root.innerHTML='<h2>ประเมินครบแล้ว</h2><p>ดาวน์โหลดไฟล์และส่งคืนผู้ดูแลการทดลอง</p><button onclick="download()">ดาวน์โหลดผล JSON</button>';return}}
 if(stage==='pre'){{root.innerHTML=`<h2>คลิป ${{caseIndex+1}} / ${{packet.cases.length}}</h2><video controls src="${{esc(c.video_uri)}}"></video><p>ก่อนเห็นคำแนะนำ คลิปนี้ยังมีโอกาสปรับเนื้อหาหรือไม่?</p><select id="opp"><option value="">เลือกคำตอบ</option><option value="yes">มี</option><option value="no">ไม่มี</option><option value="unclear">ไม่แน่ใจ</option></select><label>เหตุผล<textarea id="preNote"></textarea></label><button onclick="savePre()">ยืนยันแล้วดูคำแนะนำ</button>`;return}}
 const p=c.presentations[presentationIndex]; const empty=p.item_count===0;
 root.innerHTML=`<h2>ชุดคำแนะนำ ${{presentationIndex+1}} / ${{c.presentations.length}}</h2><p class="muted">รหัส ${{esc(p.presentation_id)}}</p>${{renderItems(p)}}<label><input id="missed" type="checkbox"> ระบบงดคำแนะนำ แต่จริง ๆ ยังมีประเด็นสำคัญที่ควรเสนอ</label>${{empty?score('abstention_appropriateness','คุณภาพการงดแนะนำ เมื่อเห็นว่าควรงด (1–5)')+score('actionability','หากเห็นว่าพลาดโอกาส ให้ Actionability = 1'):utilityFields.map(f=>score(f,({{relevance:'ความเกี่ยวข้อง',novelty:'ไม่ซ้ำสิ่งที่พูดแล้ว',clarity:'เข้าใจง่าย',actionability:'นำไปทำต่อได้'}}[f]))).join('')}}${{p.evidence_score_applicable?score('evidence_correctness','ความถูกต้องของหลักฐาน'):''}}<label>เหตุผลเมื่อให้คะแนน 1–2<textarea id="lowReason"></textarea></label><fieldset><legend>ข้อผิดพลาดร้ายแรง (ถ้ามี)</legend>${{criticalFlags.map(f=>`<label><input type="checkbox" name="flag" value="${{f}}"> ${{f}}</label>`).join('')}}</fieldset><button onclick="saveRating()">บันทึกชุดนี้</button>`}}
function start(){{if(!document.getElementById('consent').checked)return alert('ต้องยืนยันความยินยอมก่อน');response.consent=true;stage='pre';render()}}
function savePre(){{const v=document.getElementById('opp').value;if(!v)return alert('กรุณาตอบคำถามก่อน');const c=packet.cases[caseIndex];response.pre_reviews.push({{case_id:c.case_id,improvement_opportunity:v,notes:document.getElementById('preNote').value}});stage='rating';render()}}
function saveRating(){{const c=packet.cases[caseIndex],p=c.presentations[presentationIndex],scores={{}},empty=p.item_count===0,missed=document.getElementById('missed').checked;
 const names=empty?(missed?['actionability']:['abstention_appropriateness']):utilityFields.concat(p.evidence_score_applicable?['evidence_correctness']:[]);
 for(const name of names){{const el=document.getElementById(name);if(!el||!el.value)return alert('กรุณาให้คะแนนทุกช่องที่เกี่ยวข้อง');scores[name]=Number(el.value)}}
 if(!p.evidence_score_applicable)scores.evidence_correctness='N/A';if(!empty)scores.abstention_appropriateness='N/A';
 const low=Object.values(scores).some(v=>Number.isInteger(v)&&v<=2),reason=document.getElementById('lowReason').value.trim();if(low&&!reason)return alert('กรุณาระบุเหตุผลของคะแนน 1–2');
 response.ratings.push({{case_id:c.case_id,presentation_id:p.presentation_id,unit_id:p.unit_id,output_sha256:p.output_sha256,scores,missed_opportunity:missed,low_score_reason:reason,critical_flags:[...document.querySelectorAll('input[name=flag]:checked')].map(x=>x.value)}});
 presentationIndex++;if(presentationIndex>=c.presentations.length){{caseIndex++;presentationIndex=0;stage='pre'}}render()}}
function download(){{const b=new Blob([JSON.stringify(response,null,2)],{{type:'application/json'}}),a=document.createElement('a');a.href=URL.createObjectURL(b);a.download=`${{packet.study_id}}-${{packet.reviewer_id}}-responses.json`;a.click();URL.revokeObjectURL(a.href)}}render();
</script></body></html>"""


def audit_actionable_evidence(result: dict) -> list[dict]:
    """Check frozen actionable evidence without claiming semantic usefulness."""
    recommendation = result.get("recommendation") or result
    actionable = recommendation.get("actionable_recommendations") or {}
    bundle = recommendation.get("evidence_bundle") or {}
    topics = {str(row.get("topic_id")): row for row in bundle.get("action_topics", [])
              if isinstance(row, dict)}
    documents = {row.get("dataset_id"): row for row in bundle.get("reference_documents", [])
                 if isinstance(row, dict)}
    comparisons = {str(row.get("evidence_topic_id")): row
                   for row in (bundle.get("topic_comparisons") or {}).get("items", [])
                   if isinstance(row, dict)}
    checks = []
    for item in actionable.get("items", []):
        issues = []
        topic_id = str(item.get("evidence_topic_id") or "")
        topic = topics.get(topic_id)
        if not topic:
            issues.append("missing_action_topic")
            topic = {}
        item_ids = list(item.get("supporting_dataset_row_ids") or [])
        references = list(topic.get("references") or [])
        reference_ids = [row.get("dataset_id") for row in references]
        if len(item_ids) != len(set(item_ids)) or set(item_ids) != set(reference_ids):
            issues.append("supporting_rows_do_not_match_topic")
        if item.get("support_count") != len(set(reference_ids)):
            issues.append("invalid_action_support_count")
        channels = set()
        for reference in references:
            dataset_id = reference.get("dataset_id")
            document = documents.get(dataset_id)
            if not document or document.get("data_split") != "train":
                issues.append("invalid_action_reference_row")
                continue
            if document.get("channel_id"):
                channels.add(document["channel_id"])
            transcript = str(document.get("transcript") or "")
            for occurrence in reference.get("occurrences") or []:
                quote = str(occurrence.get("quote") or "")
                start, end = occurrence.get("quote_start_char"), occurrence.get("quote_end_char")
                if not isinstance(start, int) or not isinstance(end, int) or transcript[start:end] != quote:
                    issues.append("fabricated_or_misaligned_quote")
                timestamp = occurrence.get("timestamp")
                if timestamp is not None and not isinstance(timestamp, (int, float, dict, list)):
                    issues.append("invalid_timestamp")
            for field in ("published_at", "statistics_captured_at", "video_id", "channel_id"):
                if not document.get(field):
                    issues.append(f"missing_reference_{field}")
        if item.get("channel_count") != len(channels):
            issues.append("invalid_action_channel_count")
        if item.get("causal_engagement_claim") is not False:
            issues.append("causal_engagement_claim")
        if topic_id and topic_id not in comparisons:
            issues.append("missing_topic_comparison_snapshot")
        checks.append({"advice_id": item.get("id"), "evidence_topic_id": topic_id,
                       "structural_issues": sorted(set(issues)),
                       "human_review": "pending"})
    return checks


def evidence_correctness_summary(case_outputs: dict) -> dict:
    primary_cases = [case for case in case_outputs.values()
                     if case.get("role") == "heldout" and case.get("manifest_eligible")]
    regression_cases = [case for case in case_outputs.values() if case.get("role") == "regression"]
    keyword_checks = [check for case in primary_cases
                      for check in case.get("recommendation_checks", [])]
    action_checks = [check for case in primary_cases
                     for check in case.get("actionable_evidence_checks", [])]
    regression_checks = [check for case in regression_cases
                         for lane in ("recommendation_checks", "actionable_evidence_checks")
                         for check in case.get(lane, [])]
    checks = keyword_checks + action_checks
    issues = {}
    for check in checks:
        for issue in check.get("structural_issues", []):
            issues[issue] = issues.get(issue, 0) + 1
    return {
        "scope": "automated traceability and arithmetic checks; not human utility",
        "case_count": len(case_outputs),
        "primary_case_count": len(primary_cases),
        "regression_case_count": len(regression_cases),
        "recommendation_unit_count": len(checks),
        "keyword_unit_count": len(keyword_checks),
        "actionable_unit_count": len(action_checks),
        "units_with_automated_issues": sum(bool(row.get("structural_issues")) for row in checks),
        "issues": issues,
        "regression_unit_count": len(regression_checks),
        "regression_units_with_automated_issues": sum(
            bool(row.get("structural_issues")) for row in regression_checks),
        "human_evidence_correctness": "reported separately from reviewer ratings",
    }


def thai_report(*, status: str, classification: dict, evidence: dict, utility: dict,
                readiness: dict, validated: dict) -> str:
    missing = readiness.get("missing") or []
    lines = [
        "# รายงานประเมินประโยชน์คำแนะนำ Phase 7", "",
        "## สถานะ", "", f"`{status}`", "",
        "รายงานแยกความแม่นการจำแนกหมวด ความถูกต้องของหลักฐาน และประโยชน์ของคำแนะนำออกจากกัน",
        "ผลนี้ไม่ใช่หลักฐานว่าคำแนะนำทำให้ยอดวิว ไลก์ หรือความคิดเห็นเพิ่มขึ้น", "",
        "## ความพร้อมของ Pilot", "",
    ]
    if missing:
        lines.extend(f"- `{row['requirement']}`: มี {row['available']} / ต้องการ {row['required']}" for row in missing)
    else:
        lines.append("- Manifest และแผนผู้ประเมินครบตาม Pilot ที่ลงทะเบียน")
    lines.extend([
        "", "## 1. การจำแนกหมวด", "",
        f"- คลิปที่ลงทะเบียนและเข้าเกณฑ์: {classification.get('registered_eligible_count', 0)}",
        f"- วิเคราะห์สำเร็จ: {classification.get('successful_count', 0)}",
        f"- งานล้มเหลว: {classification.get('failure_count', 0)} (ไม่นับเป็นการปฏิเสธ Unknown)",
        f"- Accuracy แบบปลายทาง: {classification.get('end_to_end_accuracy')}",
        f"- Unknown false acceptance: {classification.get('unknown_false_acceptance')}",
        "", "## 2. ความถูกต้องของหลักฐาน", "",
        f"- หน่วยคำแนะนำที่ตรวจอัตโนมัติ: {evidence.get('recommendation_unit_count', 0)}",
        f"- หน่วยที่พบปัญหา: {evidence.get('units_with_automated_issues', 0)}",
        "- การตรวจอัตโนมัติยืนยันการย้อนแถวข้อมูลและสูตร ไม่ใช้แทนคะแนนความเป็นประโยชน์จากคน",
        "", "## 3. ประโยชน์ของคำแนะนำ", "",
        f"- สถานะ: `{utility.get('status')}`",
        f"- คะแนนที่รับได้: {validated.get('accepted_rating_count', 0)}",
        f"- ผู้ประเมินที่มีคะแนนรับได้: {validated.get('accepted_reviewer_count', 0)}",
        f"- แถวที่ปฏิเสธ: {len(validated.get('rejected', []))}",
        "- คะแนนที่ขาดไม่ถูกแทนด้วยศูนย์ และ A/B ไม่ถูกบังคับให้มีคะแนนหลักฐาน",
        "", "## ขอบเขตข้อสรุป", "",
        "- Classification confidence ไม่ใช่คะแนนคุณภาพคำแนะนำ",
        "- ความสัมพันธ์ใน Dataset ไม่ใช่เหตุและผล",
        "- ต้องมีคลิปใหม่และคะแนนผู้ประเมินจริงครบก่อนใช้คำว่า evaluation_complete",
    ])
    return "\n".join(lines) + "\n"

