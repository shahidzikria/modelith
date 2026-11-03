import { expect, test, type Page } from '@playwright/test';

async function openStudio(page: Page) {
  await page.goto('/#/preprocess');
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await expect(page.getByRole('img', { name: /result/ })).toBeVisible({ timeout: 30_000 });
  // The default pipeline (ImageNet classification) has finished running when the timing shows img/s.
  await expect(page.getByText(/img\/s/)).toBeVisible({ timeout: 30_000 });
}

const steps = (page: Page) => page.getByRole('list', { name: 'Pipeline steps' }).locator('[data-step-row]');
const library = (page: Page, label: string) => page.getByRole('navigation', { name: 'Step library' }).getByRole('button', { name: label, exact: true });

test('studio loads the default pipeline and previews every step', async ({ page }) => {
  await openStudio(page);
  await expect(steps(page)).toHaveCount(5);
  await expect(steps(page).first()).toContainText('Resize');
  await expect(page.getByRole('listbox', { name: /Filmstrip/ }).getByRole('option')).toHaveCount(6);
  await expect(page.getByText('3×224×224 float32').first()).toBeVisible();
});

async function dragTo(page: Page, from: { x: number; y: number }, to: { x: number; y: number }) {
  await page.mouse.move(from.x, from.y);
  await page.mouse.down();
  await page.mouse.move(from.x + 10, from.y - 10, { steps: 4 });
  // Up / down first, then across, so the pointer does not linger on the pipeline's edges (auto-scroll).
  await page.mouse.move(from.x + 10, to.y, { steps: 10 });
  await page.mouse.move(to.x, to.y, { steps: 10 });
  await page.mouse.up();
}

const centre = (b: { x: number; y: number; width: number; height: number }) => ({ x: b.x + b.width / 2, y: b.y + b.height / 2 });

test('drag steps from the library into the pipeline', async ({ page }) => {
  await openStudio(page);
  await page.getByLabel('Search steps').fill('gaussian');
  // Onto the top half of the first row: inserted before it.
  const first = (await steps(page).first().boundingBox())!;
  await dragTo(page, centre((await library(page, 'Gaussian blur').boundingBox())!), { x: first.x + first.width / 2, y: first.y + 6 });
  await expect(steps(page)).toHaveCount(6);
  await expect(steps(page).first()).toContainText('Gaussian blur');
  // Onto the "+ add step" zone: appended.
  await page.getByLabel('Search steps').fill('median');
  const end = page.getByRole('button', { name: '+ add step' });
  await end.scrollIntoViewIfNeeded();
  await dragTo(page, centre((await library(page, 'Median blur').boundingBox())!), centre((await end.boundingBox())!));
  await expect(steps(page)).toHaveCount(7);
  await expect(steps(page).last()).toContainText('Median blur');
});

test('click to append, fix a validation error by reordering (mouse and keyboard), undo', async ({ page }) => {
  await openStudio(page);
  await library(page, 'CLAHE').click();
  await expect(steps(page)).toHaveCount(6);
  await expect(steps(page).last()).toContainText('CLAHE');
  // CLAHE after Standardise is an error with a one-click fix.
  await expect(steps(page).last().getByLabel('error')).toBeVisible();

  // Keyboard: Alt+↑ moves the focused step up.
  await steps(page).last().focus();
  for (let i = 0; i < 4; i++) await page.keyboard.press('Alt+ArrowUp');
  await expect(steps(page).nth(1)).toContainText('CLAHE');
  await expect(steps(page).nth(1).getByLabel('error')).toHaveCount(0);

  // Mouse: drag CLAHE's ⋮⋮ handle onto the first row.
  await steps(page).first().scrollIntoViewIfNeeded();
  const handle = page.getByRole('button', { name: /Drag to reorder CLAHE/ });
  await dragTo(page, centre((await handle.boundingBox())!), centre((await steps(page).first().boundingBox())!));
  await expect(steps(page).first()).toContainText('CLAHE');

  // Undo restores the previous order.
  await page.keyboard.press('Control+z');
  await expect(steps(page).nth(1)).toContainText('CLAHE');
});

test('toggle a step, edit a parameter, see shapes and code update', async ({ page }) => {
  await openStudio(page);
  await steps(page).first().click(); // Resize
  await page.getByLabel('Mode', { exact: true }).selectOption('exact');
  await page.getByLabel('Size (W × H) width').fill('300');
  await page.getByLabel('Size (W × H) height').fill('200');
  await expect(steps(page).first()).toContainText('200×300×3 uint8');

  await steps(page).nth(1).getByRole('checkbox').uncheck(); // disable the centre crop
  await expect(page.getByText('Output (inference)')).toContainText('3×200×300');

  await page.getByRole('tab', { name: 'Code' }).click();
  await expect(page.locator('.monaco-editor')).toContainText('Image preprocessing pipeline', { timeout: 20_000 });
  await page.getByRole('button', { name: 'Copy file' }).click();
  // Monaco only renders visible lines, so check the whole file through the clipboard.
  await expect.poll(() => page.evaluate(() => navigator.clipboard.readText())).toContain('RESIZE_W = 300');
  const clip = await page.evaluate(() => navigator.clipboard.readText());
  expect(clip).not.toContain('def step_02_crop');
  expect(clip).toContain('def preprocess(img, mask=None, boxes=None, keypoints=None, train=False, rng=None');
});

test('presets, train / inference mode and 9 random samples', async ({ page }) => {
  await openStudio(page);
  await page.getByRole('button', { name: 'Presets' }).click();
  await page.getByRole('menuitem', { name: /ImageNet training augmentation/ }).click();
  await expect(steps(page)).toHaveCount(8);
  await page.getByRole('radio', { name: 'Train' }).click();
  await expect(page.getByText('Output (train)')).toBeVisible();
  await steps(page).nth(3).click(); // Random flip
  await page.getByRole('button', { name: /Show 9 samples/ }).click();
  await expect(page.getByRole('dialog', { name: '9 random samples' }).getByRole('img')).toHaveCount(9, { timeout: 30_000 });
});

test('variant B compare view', async ({ page }) => {
  await openStudio(page);
  await page.getByRole('button', { name: /variant B/ }).click();
  await expect(page.getByText('Comparing variant A and B')).toBeVisible();
  await expect(page.getByRole('img', { name: 'Difference A − B' })).toBeVisible({ timeout: 30_000 });
});

test('linking a pipeline to the model shows a badge and feeds train.py', async ({ page }) => {
  await page.goto('/');
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await page.getByRole('button', { name: /Simple CNN classifier/ }).click();
  await page.locator('.react-flow__node').filter({ hasText: 'Image' }).first().click({ position: { x: 60, y: 14 } });
  // No pipelines yet until the studio has been opened once: open it, then come back.
  await page.getByRole('link', { name: /Open Preprocessing Studio/ }).click();
  await expect(page.getByText(/img\/s/)).toBeVisible({ timeout: 30_000 });
  await page.getByLabel('Link to model').selectOption({ index: 1 });
  await page.getByRole('link', { name: 'Model' }).click();
  await expect(page.locator('.react-flow__node').filter({ hasText: '🔗' })).toBeVisible();
  await page.getByRole('button', { name: /Generate code/ }).click();
  await expect(page.getByRole('tab', { name: 'preprocess.py' })).toBeVisible({ timeout: 20_000 });
  await page.getByRole('tab', { name: 'train.py' }).click();
  await expect(page.locator('.monaco-editor')).toContainText('import dataset as D', { timeout: 20_000 });
});

test('tablet width (768 px): library in a drawer, pipeline beside the viewer', async ({ page }) => {
  await page.setViewportSize({ width: 768, height: 1024 });
  await openStudio(page);
  await expect(page.getByRole('navigation', { name: 'Step library' })).toBeHidden();
  await page.getByRole('button', { name: 'Open step library' }).click();
  await library(page, 'Median blur').click();
  await expect(steps(page)).toHaveCount(6);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(0);
});
