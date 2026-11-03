import { select } from '../../params';
import { defineStep } from '../../registry';

const EFFECTS: Record<string, { label: string; a: string }> = {
  fog: { label: 'Fog', a: 'A.RandomFog(fog_coef_range=(0.3, 0.6), alpha_coef=0.08, p=1.0)' },
  rain: { label: 'Rain', a: 'A.RandomRain(slant_range=(-10, 10), drop_length=15, blur_value=3, brightness_coefficient=0.8, p=1.0)' },
  shadow: { label: 'Shadow', a: 'A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_limit=(1, 2), p=1.0)' },
  sun_flare: { label: 'Sun flare', a: 'A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), src_radius=120, p=1.0)' },
  snow: { label: 'Snow', a: 'A.RandomSnow(snow_point_range=(0.1, 0.3), brightness_coeff=2.0, p=1.0)' },
};

defineStep({
  id: 'weather',
  category: 'Augmentation',
  label: 'Weather & light (Albumentations)',
  doc: 'Synthetic fog, rain, snow, cast shadows or sun flare from Albumentations, for outdoor / driving datasets. The Albumentations RNG is seeded from the pipeline RNG, so results are repeatable. Needs albumentations (optional dependency) and runs on the server.',
  cv2: 'A.RandomFog / RandomRain / RandomShadow / RandomSunFlare / RandomSnow',
  keywords: 'weather fog rain snow shadow sun flare albumentations driving outdoor augmentation',
  params: [select('effect', 'Effect', Object.entries(EFFECTS).map(([k, v]) => [k, v.label] as [string, string]), 'fog')],
  engine: 'server',
  kind: 'random',
  affectsGeometry: false,
  accepts: { c: [3], dtype: ['uint8'], space: ['RGB', 'BGR'] },
  summary: (p) => EFFECTS[p.effect]?.label ?? p.effect,
  outMeta: (m) => m,
  emitPython: (p, ctx) => {
    ctx.helper('randint');
    const rgb = ctx.in.space === 'RGB';
    return {
      body: [
        'import albumentations as A  # optional dependency',
        '',
        `aug = A.Compose([${EFFECTS[p.effect].a}], seed=_randint(rng, 0, 2**31 - 1))`,
        rgb ? 'img = aug(image=img)["image"]' : 'img = cv2.cvtColor(aug(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))["image"], cv2.COLOR_RGB2BGR)  # Albumentations expects RGB',
      ].join('\n'),
    };
  },
  emitAlbumentations: (p) => EFFECTS[p.effect].a.replace('p=1.0', 'p={p}'),
  testCases: [{ seed: 211 }, { seed: 212, params: { effect: 'rain' } }],
});
