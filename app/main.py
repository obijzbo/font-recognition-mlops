from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tasks import preprocess_and_predict

app = FastAPI()


@app.post("/classify-image")
async def classify_image(file: UploadFile = File(...)):
    """
    Accept an image, enqueue preprocessing and prediction, and return task ID.
    """
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image format. Use JPEG or PNG.")

    try:
        # Read the image data
        image_data = await file.read()

        # Trigger the Celery task
        task = preprocess_and_predict.delay(image_data)

        # Return the task ID for tracking
        return {"task_id": task.id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/result/{task_id}")
def get_result(task_id: str):
    """
    Fetch the result of the task using task ID.
    """
    task_result = preprocess_and_predict.AsyncResult(task_id)

    if task_result.state == "PENDING":
        return {"task_id": task_id, "status": "PENDING"}

    if task_result.state == "FAILURE":
        return {"task_id": task_id, "status": "FAILED", "error": str(task_result.result)}

    return {"task_id": task_id, "status": "SUCCESS", "result": task_result.result}
