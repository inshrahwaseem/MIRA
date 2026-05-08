"""Tests for MIRA Psychology Module — theory engine, prescriptions, crisis protocol."""
import pytest

class TestMaslowAnalyzer:
    def test_physiological_detection(self):
        from app.psychology.theory_engine import MaslowNeedsAnalyzer, MaslowLevel
        a = MaslowNeedsAnalyzer()
        level, conf = a.analyze_need_level("I'm exhausted, can't sleep, no energy at all")
        assert level == MaslowLevel.PHYSIOLOGICAL

    def test_belonging_detection(self):
        from app.psychology.theory_engine import MaslowNeedsAnalyzer, MaslowLevel
        a = MaslowNeedsAnalyzer()
        level, _ = a.analyze_need_level("I feel so lonely, nobody cares, I'm isolated")
        assert level == MaslowLevel.BELONGING

    def test_esteem_detection(self):
        from app.psychology.theory_engine import MaslowNeedsAnalyzer, MaslowLevel
        a = MaslowNeedsAnalyzer()
        level, _ = a.analyze_need_level("I'm a failure, not good enough, everyone is better")
        assert level == MaslowLevel.ESTEEM

    def test_interventions_exist(self):
        from app.psychology.theory_engine import MaslowNeedsAnalyzer, MaslowLevel
        a = MaslowNeedsAnalyzer()
        for lv in MaslowLevel:
            interventions = a.get_intervention(lv)
            assert len(interventions) >= 2

class TestAttachmentDetector:
    def test_anxious_detection(self):
        from app.psychology.theory_engine import AttachmentStyleDetector, AttachmentStyle
        d = AttachmentStyleDetector()
        style, _ = d.infer_style(["they don't care about me", "what if they leave", "please don't go"])
        assert style == AttachmentStyle.ANXIOUS_PREOCCUPIED

    def test_avoidant_detection(self):
        from app.psychology.theory_engine import AttachmentStyleDetector, AttachmentStyle
        d = AttachmentStyleDetector()
        style, _ = d.infer_style(["i don't need anyone", "better alone", "emotions are weakness"])
        assert style == AttachmentStyle.DISMISSIVE_AVOIDANT

    def test_communication_adaptations(self):
        from app.psychology.theory_engine import AttachmentStyleDetector, AttachmentStyle
        d = AttachmentStyleDetector()
        adapt = d.adapt_communication(AttachmentStyle.ANXIOUS_PREOCCUPIED)
        assert "reassur" in adapt.tone.lower()

class TestCBTEngine:
    def test_identify_catastrophizing(self):
        from app.psychology.theory_engine import CBTInterventionEngine
        e = CBTInterventionEngine()
        nats = e.identify_nats("Everything is going to fall apart and it will be a disaster")
        assert any(n.distortion_type == "catastrophizing" for n in nats)

    def test_thought_record(self):
        from app.psychology.theory_engine import CBTInterventionEngine, NegativeAutomaticThought
        e = CBTInterventionEngine()
        nat = NegativeAutomaticThought("I always fail", "all_or_nothing", "Sadness", 0.7)
        record = e.generate_thought_record(nat)
        assert record.balanced_thought
        assert len(record.evidence_against) > 0

    def test_socratic_questions(self):
        from app.psychology.theory_engine import CBTInterventionEngine
        e = CBTInterventionEngine()
        qs = e.socratic_questions("nobody likes me")
        assert len(qs) == 5

    def test_behavioral_activation_high(self):
        from app.psychology.theory_engine import CBTInterventionEngine
        e = CBTInterventionEngine()
        plan = e.behavioral_activation_plan(0.8)
        assert len(plan) >= 4

class TestACTEngine:
    def test_defusion(self):
        from app.psychology.theory_engine import ACTTechniqueEngine
        e = ACTTechniqueEngine()
        result = e.defusion_exercise("I'm worthless")
        assert "worthless" in result.lower()

    def test_values(self):
        from app.psychology.theory_engine import ACTTechniqueEngine
        e = ACTTechniqueEngine()
        qs = e.values_clarification_prompt()
        assert len(qs) >= 4

    def test_rain(self):
        from app.psychology.theory_engine import ACTTechniqueEngine
        e = ACTTechniqueEngine()
        script = e.acceptance_script("Anxiety", 2)
        assert "RAIN" in script
        assert "Recognize" in script

class TestBigFive:
    def test_infer_neuroticism(self):
        from app.psychology.theory_engine import BigFiveProfiler
        p = BigFiveProfiler()
        scores = p.infer_from_history(["I worry all the time", "anxious about worst case", "can't stop thinking", "overwhelmed and panic"])
        assert scores.neuroticism > 0.5

    def test_adapt_low_extraversion(self):
        from app.psychology.theory_engine import BigFiveProfiler, BigFiveScores
        p = BigFiveProfiler()
        scores = BigFiveScores(extraversion=0.2, neuroticism=0.5)
        adapt = p.adapt_response_style(scores)
        assert adapt.social_suggestions is False

class TestPrescriptionDB:
    def test_total_count(self):
        from app.psychology.prescription_db import PRESCRIPTIONS
        assert len(PRESCRIPTIONS) >= 40

    def test_all_have_steps(self):
        from app.psychology.prescription_db import PRESCRIPTIONS
        for rx in PRESCRIPTIONS:
            assert len(rx.steps) >= 2, f"{rx.name} has too few steps"

    def test_selector_anxiety(self):
        from app.psychology.prescription_db import PrescriptionSelector
        s = PrescriptionSelector()
        results = s.rank_prescriptions("Anxiety", k=5)
        assert len(results) >= 3
        assert all("Anxiety" in rx.target_emotions or "anxiety" in str(rx.target_emotions).lower() for rx in results)

    def test_selector_excludes_recent(self):
        from app.psychology.prescription_db import PrescriptionSelector
        s = PrescriptionSelector()
        results = s.rank_prescriptions("Sadness", recent_prescriptions=["Behavioral Activation"], k=5)
        assert all(rx.name != "Behavioral Activation" for rx in results)

class TestCrisisKeywords:
    def test_high_risk_detected(self):
        from app.psychology.crisis_protocol import KeywordCrisisDetector, CrisisLevel
        d = KeywordCrisisDetector()
        triggered, level, _ = d.detect("I want to kill myself")
        assert level == CrisisLevel.IMMEDIATE
        assert len(triggered) >= 1

    def test_gaming_context_negated(self):
        from app.psychology.crisis_protocol import KeywordCrisisDetector, CrisisLevel
        d = KeywordCrisisDetector()
        _, level, _ = d.detect("I'm going to kill this boss level in the game")
        assert level == CrisisLevel.NONE

    def test_medium_risk(self):
        from app.psychology.crisis_protocol import KeywordCrisisDetector, CrisisLevel
        d = KeywordCrisisDetector()
        _, level, _ = d.detect("I feel hopeless and worthless")
        assert level == CrisisLevel.ELEVATED

    def test_safe_text(self):
        from app.psychology.crisis_protocol import KeywordCrisisDetector, CrisisLevel
        d = KeywordCrisisDetector()
        _, level, _ = d.detect("I had a great day today and I'm happy")
        assert level == CrisisLevel.NONE

class TestMLCrisisScorer:
    def test_very_negative_valence(self):
        from app.psychology.crisis_protocol import MLCrisisScorer, CrisisLevel
        s = MLCrisisScorer()
        level, score = s.score(valence=-0.9, has_future_references=False, keyword_count=2)
        assert level >= CrisisLevel.ELEVATED

    def test_normal_scores(self):
        from app.psychology.crisis_protocol import MLCrisisScorer, CrisisLevel
        s = MLCrisisScorer()
        level, _ = s.score(valence=0.3, arousal=0.5)
        assert level == CrisisLevel.NONE

class TestBehavioralDetector:
    def test_flat_affect(self):
        from app.psychology.crisis_protocol import BehavioralCrisisDetector
        d = BehavioralCrisisDetector()
        voice = {"pitch": {"std_f0": 5}, "energy": {"rms_mean": 0.005}, "temporal": {"speech_rate_estimate": 1.0}}
        add, flags = d.detect(voice_features=voice)
        assert add == 1
        assert any("FLAT_AFFECT" in f for f in flags)

    def test_no_features(self):
        from app.psychology.crisis_protocol import BehavioralCrisisDetector
        d = BehavioralCrisisDetector()
        add, flags = d.detect()
        assert add == 0

class TestCrisisSystem:
    def test_full_assessment_safe(self):
        from app.psychology.crisis_protocol import CrisisDetectionSystem, CrisisLevel
        s = CrisisDetectionSystem()
        result = s.assess("I feel great today", valence=0.8)
        assert result.level == CrisisLevel.NONE

    def test_full_assessment_crisis(self):
        from app.psychology.crisis_protocol import CrisisDetectionSystem, CrisisLevel
        s = CrisisDetectionSystem()
        result = s.assess("I want to end it all, no reason to live", valence=-0.9)
        assert result.level == CrisisLevel.IMMEDIATE

    def test_response_protocol_level3(self):
        from app.psychology.crisis_protocol import CrisisResponseProtocol, CrisisLevel
        resp = CrisisResponseProtocol.get_response(CrisisLevel.IMMEDIATE, 0)
        assert resp is not None
        assert "safe" in resp.lower() or "yahaan" in resp.lower()
