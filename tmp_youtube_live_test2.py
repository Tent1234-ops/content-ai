from app.services.trends import get_youtube_trending
from app.core.config import settings
import json
mode, items = get_youtube_trending(region=settings.youtube_region, limit=5, mode='live')
out = []
for it in items:
    d = it.model_dump() if hasattr(it, 'model_dump') else it.dict()
    if d.get('published_at'):
        try:
            d['published_at'] = d['published_at'].isoformat()
        except Exception:
            d['published_at'] = str(d['published_at'])
    out.append(d)
with open('tmp_youtube_live.json','w',encoding='utf-8') as f:
    json.dump({'mode':mode,'region':settings.youtube_region,'total':len(out),'items':out}, f, ensure_ascii=False, indent=2)
print('WROTE tmp_youtube_live.json')
