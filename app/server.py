import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from itertools import islice

export_file_url = 'https://drive.google.com/uc?export=download&id=1l9NpwIXLxpagEq7l0oSLK0DqEuUH5H3k'
export_file_name = 'export.pkl'

classes = ['steinsopp', 'giftslørsopp', 'sort_trompetsopp', 'seig_kusopp', 'hvit_fluesopp', 'grønn_fluesopp', 'kantarell', 'gul_trompetsopp', 'rød_fluesopp']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    
    # Prediction for one image
pred = learn_test.predict(img)
print(pred[0])
predclasses = pred[2]
p = predclasses.tolist()
dictpred = dict(zip(learn.data.classes, p))
dictpred_sorted = {k: v for k, v in sorted(dictpred.items(), reverse=True, key=lambda x: x[1])}
dictpred_sorted_top3 = list(islice(dictpred_sorted.items(), 3))
    return JSONResponse({'result': str(dictpred_sorted_top3)})
#print(dictpred_sorted_top3)
    



if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
