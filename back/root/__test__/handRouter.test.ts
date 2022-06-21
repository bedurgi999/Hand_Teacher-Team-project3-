/* eslint-disable no-undef */
import app from "../src/app";
import request from "supertest";

describe("End-to-End Test", () => {
  describe("GET /hands", () => {
    test("Get all Sign Language Data from Hand DB", async () => {
      await request(app).get("/hands").set("Accept", "application/json").expect(200).expect("Content-Type", /json/);
    });
  });

  describe("GET /:alphabet", () => {
    test("Get Sign Language Data by Alphabet", async () => {
      let param = "/b";
      await request(app)
        .get("/hands" + param)
        .set("Accept", "application/json")
        .expect(200)
        .expect("Content-Type", /json/);
    });
  });

  describe("POST /hand", () => {
    test("Post hand data", async () => {
      await request(app)
        .post("/hands")
        .set("Accept", "application/json")
        .type("application/json")
        .send({
          alphabet: "test",
          handImage: "test_hand_image",
          mouthImage: "test_mouth_image",
          video: "test_video",
        })
        .expect(200)
        .expect("Content-Type", /json/);
    });
  });
});
