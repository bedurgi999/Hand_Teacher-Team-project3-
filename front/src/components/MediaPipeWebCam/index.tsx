import React, { useRef, useEffect, useState } from "react";
import {
  Holistic,
  HAND_CONNECTIONS,
  POSE_CONNECTIONS,
} from "@mediapipe/holistic";
import * as cam from "@mediapipe/camera_utils";
import * as h from "@mediapipe/holistic";
import Webcam from "react-webcam";
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";
import { UserCanvas } from "./index.style";

const holistic = new Holistic({
  locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
  },
});
holistic.setOptions({
  modelComplexity: 1,
  smoothLandmarks: true,
  enableSegmentation: true,
  smoothSegmentation: true,
  refineFaceLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
});

interface WebCamProps {
  cameraOn: boolean;
}

function MediaPipeWebCam({ cameraOn }: WebCamProps) {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // let camera = null;
  const [camera, setCamera] = useState<cam.Camera>();
  const [test, setTest] = useState<h.NormalizedLandmarkList[]>([])

  const onResults: h.ResultsListener = (results) => {
    // console.log(results.poseLandmarks);
    if(test.length <= 30) {
      setTest((cur):h.NormalizedLandmarkList[] => {
        const temp = [...cur];
        temp.push(results.poseLandmarks);
        console.log(temp);
        
        return temp;
      })
    }
    

    
    // if (!canvasRef.current || !webcamRef.current?.video) {
    //   return;
    // }

    // canvasRef.current.width = webcamRef.current?.video.videoWidth;
    // canvasRef.current.height = webcamRef.current?.video.videoHeight;

    // const canvasElement = canvasRef.current;
    // const canvasCtx = canvasElement.getContext("2d");

    // if (!canvasCtx) {
    //   return;
    // }

    // canvasCtx.save();

    // canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    // canvasCtx.globalCompositeOperation = "source-in";
    // canvasCtx.fillStyle = "#00FF00";
    // canvasCtx.fillRect(0, 0, canvasElement.width, canvasElement.height);

    // canvasCtx.globalCompositeOperation = "destination-atop";
    // canvasCtx.drawImage(
    //   results.image,
    //   0,
    //   0,
    //   canvasElement.width,
    //   canvasElement.height
    // );
    // canvasCtx.globalCompositeOperation = "source-over";

    // drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, {
    //   color: "#00FF00",
    //   lineWidth: 4,
    // });
    // drawLandmarks(canvasCtx, results.poseLandmarks, {
    //   color: "#FF0000",
    //   lineWidth: 2,
    // });
    // drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS, {
    //   color: "#CC0000",
    //   lineWidth: 5,
    // });
    // drawLandmarks(canvasCtx, results.leftHandLandmarks, {
    //   color: "#00FF00",
    //   lineWidth: 2,
    // });
    // drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS, {
    //   color: "#00CC00",
    //   lineWidth: 5,
    // });
    // drawLandmarks(canvasCtx, results.rightHandLandmarks, {
    //   color: "#FF0000",
    //   lineWidth: 2,
    // });
  };

  useEffect(() => {
    holistic.onResults(onResults);
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null
    ) {
      if (!webcamRef.current?.video) {
        return;
      }
      const newCamera = new cam.Camera(webcamRef.current?.video, {
        onFrame: async () => {
          if (!webcamRef.current?.video) {
            return;
          }
          await holistic.send({ image: webcamRef.current?.video });
        },
        width: 640,
        height: 480,
      });
      setCamera(newCamera);
    }
  }, []);

  useEffect(() => {
    const start = async () => {
      await camera?.start();
    }
    const end = async() => {
      await camera?.stop()
    }

    try {
      start();

      return () => {
        end();
      }
    } catch(e: any) {
      throw new Error(e)
    }
  }, [cameraOn]);

  return (
    <>
      <Webcam
        ref={webcamRef}
        style={{
          position: "absolute",
          marginLeft: "auto",
          marginRight: "auto",
          textAlign: "center",
          zIndex: 9,
          width: 640,
          height: 480,
        }}
      />
      {/* {cameraOn && <UserCanvas ref={canvasRef} />} */}
    </>
  );
}

export default MediaPipeWebCam;
