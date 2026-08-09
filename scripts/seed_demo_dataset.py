from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.database.db import Base, SessionLocal, engine
from app.database.models import DatasetContent, SystemLog


def ensure_sqlite_demo_columns() -> None:
    db = SessionLocal()
    try:
        bind = db.get_bind()
        if bind.dialect.name != "sqlite":
            return
        connection = db.connection()
        rows = connection.exec_driver_sql("PRAGMA table_info('system_configs')").fetchall()
        columns = {row[1] for row in rows}
        additions = [
            ("tiktok_region", "TEXT NOT NULL DEFAULT 'TH'"),
            ("enable_tiktok_trending", "INTEGER NOT NULL DEFAULT 1"),
            ("asr_model_default", "TEXT NOT NULL DEFAULT 'small'"),
            ("enable_model_toggle", "INTEGER NOT NULL DEFAULT 1"),
            ("job_backend", "TEXT NOT NULL DEFAULT 'inprocess'"),
            ("redis_url", "TEXT NULL"),
        ]
        for column_name, sql_type in additions:
            if column_name not in columns:
                connection.exec_driver_sql(
                    f"ALTER TABLE system_configs ADD COLUMN {column_name} {sql_type}"
                )
        db.commit()
    finally:
        db.close()


DOMAIN_SEEDS = {
    "smartphone": [
        ("Camera battle under low light", "camera quality portrait mode night mode stabilization dynamic range color science chipset snapdragon battery life display quality fast charge thermal performance"),
        ("Battery drain test for creators", "battery life screen on time fast charge charger heat management chipset performance oled display refresh rate camera sample value for money"),
        ("Budget phone gaming review", "chip performance snapdragon dimensity frame rate thermal throttling battery life touch sampling display brightness stereo speaker camera quality"),
        ("Flagship camera tips", "camera quality optical stabilization portrait video stabilization microphone quality display quality battery life fast charge night mode"),
        ("Midrange phone buying guide", "value for money camera quality battery life display quality chipset performance software update fast charge build quality"),
        ("Creator phone comparison", "front camera video stabilization microphone quality color science battery life display brightness storage performance fast charge"),
        ("Compact phone daily use", "one hand comfort battery life camera quality oled display speaker quality chipset performance heat management software update"),
        ("Charging speed myth test", "fast charge battery health charger watt heat management screen on time chipset efficiency display quality value for money"),
        ("Android vs iPhone camera", "camera quality portrait mode video stabilization dynamic range color science night mode battery life display quality"),
        ("Best phone for students", "value for money battery life camera quality display quality durability fast charge chipset performance storage"),
        ("Phone display explained", "display quality oled amoled 120hz brightness touch response battery life color accuracy camera quality chipset performance"),
        ("Travel smartphone setup", "camera quality ultra wide stabilization battery life fast charge storage gps durability display brightness microphone quality"),
    ],
    "food_drink": [
        ("Street food under budget", "taste profile crispy texture portion size value for money spicy sauce freshness service quality location delivery packaging"),
        ("Cafe menu worth trying", "taste profile aroma texture sweetness balance portion size value for money presentation service quality atmosphere"),
        ("Noodle shop comparison", "taste profile broth richness noodle texture portion size value for money freshness service speed location"),
        ("Dessert cafe review", "flavor balance sweetness texture creaminess portion size presentation value for money service quality atmosphere"),
        ("Delivery food honest test", "delivery packaging freshness taste profile texture portion size value for money reheating quality service speed"),
        ("Bubble tea ranking", "taste profile tea aroma sweetness level texture pearls freshness portion size value for money service speed"),
        ("Hidden restaurant review", "taste profile signature menu freshness portion size value for money service quality atmosphere location cleanliness"),
        ("Spicy menu challenge", "spicy level flavor balance texture freshness portion size value for money sauce aroma service quality"),
        ("Breakfast cafe guide", "taste profile coffee aroma texture portion size value for money service quality atmosphere freshness"),
        ("Thai dessert explain", "sweetness balance coconut aroma texture freshness presentation portion size value for money cultural story"),
        ("Healthy lunch review", "freshness taste profile nutrition balance portion size texture value for money delivery packaging service quality"),
        ("Grilled chicken comparison", "smoky flavor texture tenderness sauce freshness portion size value for money service speed location"),
    ],
    "skincare": [
        ("Barrier repair routine", "skin barrier ceramide hydration moisturizer sensitive skin fragrance free alcohol free texture absorption irritation test"),
        ("Sunscreen for oily skin", "spf pa sunscreen texture white cast oil control sensitive skin fragrance free reapply hydration finish"),
        ("Niacinamide serum review", "niacinamide brightening oil control pores hydration sensitive skin texture absorption fragrance free irritation test"),
        ("Retinol beginner guide", "retinol active ingredients irritation test skin barrier moisturizer sensitive skin routine frequency hydration sunscreen"),
        ("Cleanser comparison", "cleanser ph balance fragrance free alcohol free sensitive skin hydration skin barrier texture residue acne prone"),
        ("Vitamin C serum test", "vitamin c brightening antioxidant texture absorption sensitive skin irritation test packaging stability sunscreen"),
        ("Acne skincare routine", "salicylic acid bha oil control acne prone skin barrier moisturizer fragrance free irritation test hydration"),
        ("Dry skin moisturizer", "moisturizer hydration ceramide skin barrier texture absorption fragrance free sensitive skin overnight repair"),
        ("Exfoliating toner review", "aha bha exfoliation texture irritation test sensitive skin frequency skin barrier hydration sunscreen"),
        ("Drugstore skincare ranking", "value for money moisturizer sunscreen cleanser niacinamide hydration fragrance free sensitive skin texture"),
        ("Sensitive skin patch test", "sensitive skin irritation test fragrance free alcohol free skin barrier hydration moisturizer texture redness"),
        ("Hyaluronic serum routine", "hyaluronic hydration plumping texture absorption moisturizer skin barrier sensitive skin fragrance free"),
    ],
    "audio": [
        ("Budget earbuds mic test", "microphone quality noise cancelling latency sound quality bass clarity battery life wear comfort gaming audio"),
        ("Gaming headset comparison", "gaming audio soundstage microphone quality latency positional audio wear comfort build quality noise isolation"),
        ("ANC earbuds daily use", "noise cancelling transparency mode sound quality microphone quality battery life wear comfort latency app control"),
        ("IEM for beginners", "sound quality soundstage bass mids treble detail imaging cable comfort value for money fit isolation"),
        ("Wireless headset latency test", "latency gaming audio microphone quality battery life wireless stability sound quality wear comfort dongle"),
        ("Podcast microphone check", "microphone quality noise rejection clarity plosive handling room noise monitoring sound quality setup"),
        ("Music earbuds ranking", "sound quality bass clarity soundstage vocal detail battery life wear comfort noise cancelling value for money"),
        ("Open ear audio review", "wear comfort awareness sound quality microphone quality battery life fit stability latency water resistance"),
        ("Headphone comfort test", "wear comfort clamping force ear pad heat sound quality soundstage build quality cable weight"),
        ("Bluetooth speaker review", "sound quality bass clarity loudness battery life water resistance latency microphone quality portability"),
        ("Creator audio kit", "microphone quality monitoring latency noise cancelling sound quality portability battery life connection stability"),
        ("Call quality showdown", "microphone quality noise rejection wind noise clarity battery life wear comfort bluetooth stability"),
    ],
    "keyboard": [
        ("Mechanical keyboard switch guide", "mechanical keyboard switch type linear tactile clicky actuation force typing feel sound profile"),
        ("Gasket keyboard sound test", "gasket mount sound profile foam dampening build quality flex typing feel stabilizer tuning"),
        ("Hot swap keyboard review", "hot swappable switch compatibility keycap profile layout build quality typing feel software tuning"),
        ("Wireless keyboard latency test", "wireless bluetooth latency connectivity battery life polling rate typing feel gaming performance"),
        ("Budget keyboard comparison", "mechanical keyboard value for money build quality switch type keycap quality sound profile"),
        ("Keyboard layout buying guide", "keyboard layout 60 percent 75 percent tkl full size productivity typing feel connectivity"),
        ("Custom keyboard beginner build", "mechanical keyboard pcb plate stabilizer switch type keycap gasket build quality sound profile"),
        ("Low profile keyboard review", "low profile switch type typing feel portability wireless battery life build quality layout"),
        ("Gaming keyboard response test", "gaming keyboard latency polling rate switch type rapid trigger software tuning build quality"),
        ("Office keyboard comfort test", "typing feel wrist comfort sound profile layout wireless connectivity battery life build quality"),
        ("Keyboard keycap comparison", "keycap material pbt abs profile texture durability sound profile typing feel compatibility"),
        ("Silent keyboard setup", "silent switch sound profile foam dampening stabilizer typing feel build quality office keyboard"),
    ],
    "mouse": [
        ("Gaming mouse sensor test", "gaming mouse sensor performance dpi tracking stability polling rate click latency software tuning"),
        ("Lightweight mouse comparison", "gaming mouse weight ergonomic shape sensor performance click latency build quality grip style"),
        ("Wireless mouse battery review", "wireless mouse battery life connectivity latency sensor performance charging build quality"),
        ("Mouse shape buying guide", "ergonomic shape palm grip claw grip fingertip grip weight comfort sensor performance"),
        ("Budget gaming mouse review", "gaming mouse value for money dpi sensor performance polling rate click latency build quality"),
        ("Mouse polling rate explained", "polling rate latency sensor performance tracking stability cpu usage gaming mouse dpi"),
        ("Productivity mouse workflow", "wireless mouse bluetooth buttons software tuning ergonomic shape battery life connectivity"),
        ("Esports mouse click test", "gaming mouse click latency switch durability sensor performance polling rate weight grip style"),
        ("Travel mouse review", "wireless mouse portability bluetooth battery life compact ergonomic shape tracking stability"),
        ("Mouse feet and pad test", "tracking stability sensor performance mouse feet glide control dpi polling rate gaming mouse"),
        ("Vertical mouse comfort review", "ergonomic shape wrist comfort vertical mouse wireless connectivity battery life build quality"),
        ("Mouse software setup", "software tuning dpi profiles macros polling rate sensor performance wireless battery life"),
    ],
    "fashion": [
        ("Everyday shirt material test", "fashion material quality cotton fabric fit size accuracy comfort durability care requirement"),
        ("Sneaker comfort comparison", "fashion sneaker fit size accuracy comfort material quality durability style versatility"),
        ("Office outfit buying guide", "fashion outfit fit material quality comfort size accuracy style versatility value for money"),
        ("Denim durability review", "fashion denim fabric material quality fit size accuracy durability care requirement comfort"),
        ("Online clothing size test", "fashion clothing size accuracy fit return policy material quality comfort value for money"),
        ("Travel outfit capsule", "fashion outfit style versatility material quality comfort durability care requirement fit"),
        ("Budget bag quality check", "fashion bag material quality stitching durability capacity comfort style versatility value for money"),
        ("Summer dress fabric review", "fashion dress fabric breathability material quality fit size accuracy comfort care requirement"),
        ("Jacket construction test", "fashion jacket material quality stitching fit durability comfort care requirement style versatility"),
        ("Activewear performance review", "fashion activewear fabric stretch breathability fit size accuracy comfort durability"),
        ("Sustainable fashion guide", "fashion material quality durability care requirement fit style versatility value for money"),
        ("Accessory styling comparison", "fashion accessory material quality comfort durability style versatility outfit fit"),
    ],
}

SEED_VARIATIONS = [
    (
        "real-world test",
        "real world usage benchmark strengths weaknesses buying advice audience comparison",
    ),
    (
        "creator comparison",
        "hook demonstration side by side evidence pros cons value for money recommendation",
    ),
]


def _trend_score(views: int, likes: int, comments: int) -> float:
    return round((views * 0.55) + (likes * 5.5) + (comments * 12.0), 2)


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    now = datetime.utcnow()
    for domain_index, (domain, items) in enumerate(DOMAIN_SEEDS.items()):
        for item_index, (title, transcript) in enumerate(items):
            for variation_index, (variation_title, variation_text) in enumerate(SEED_VARIATIONS):
                row_index = (item_index * len(SEED_VARIATIONS)) + variation_index
                views = 120_000 + (domain_index * 35_000) + (row_index * 14_500)
                likes = 7_500 + (domain_index * 1_200) + (row_index * 680)
                comments = 420 + (domain_index * 80) + (row_index * 38)
                if domain == "food_drink":
                    duration = 34 + (row_index % 6) * 6
                elif domain == "smartphone":
                    duration = 78 + (row_index % 6) * 12
                elif domain == "skincare":
                    duration = 58 + (row_index % 6) * 8
                else:
                    duration = 62 + (row_index % 6) * 10

                rows.append(
                    {
                        "title": f"{title}: {variation_title}",
                        "video_url": f"seed://youtube/{domain}/{row_index + 1:02d}",
                        "transcript": f"{transcript} {variation_text}",
                        "category": domain,
                        "source_platform": "youtube_seed",
                        "views": views,
                        "likes": likes,
                        "comments": comments,
                        "trend_score": _trend_score(views, likes, comments),
                        "duration_seconds": duration,
                        "published_at": now - timedelta(days=row_index + domain_index),
                    }
                )
    return rows


def seed_demo_dataset() -> dict[str, int]:
    Base.metadata.create_all(bind=engine)
    ensure_sqlite_demo_columns()
    db = SessionLocal()
    created = 0
    updated = 0
    try:
        for payload in build_rows():
            existing = (
                db.query(DatasetContent)
                .filter(DatasetContent.video_url == payload["video_url"])
                .first()
            )
            if existing:
                for key, value in payload.items():
                    setattr(existing, key, value)
                updated += 1
            else:
                db.add(DatasetContent(**payload))
                created += 1

        db.add(
            SystemLog(
                user_id=None,
                action="seed_demo_dataset",
                status="success",
                detail=f"created={created}, updated={updated}",
            )
        )
        db.commit()
        return {"created": created, "updated": updated, "total": created + updated}
    finally:
        db.close()


if __name__ == "__main__":
    result = seed_demo_dataset()
    print(result)
