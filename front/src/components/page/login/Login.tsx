import React, { useState, useEffect } from "react";
import {
  LoginBackground,
  LoginPage,
  LoginForm,
  LoginInput,
  InputBox,
  LoginButton,
  ValidWord,
} from "./login.style";
import loginImg from "./login.jpg";
import { countAtom } from "../../../state";
import { useAtom, Provider, atom, useAtomValue } from "jotai";
//atomwithstorage
interface UserLogin {
  id: string;
  password: string;
}

interface LoginValid {
  idValid: boolean;
  pwValid: boolean;
}

function Login() {
  // const [count, setCount] = useAtom(countAtom);
  const count = atom((get) => {
    const num = get(countAtom);
    return num;
  });
  const count2 = useAtomValue(count);
  const reg =
    /^(([^<>()[\]\\.,;:\s@"]+(\.[^<>()[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;
  const [loginInfo, setLoginInfo] = useState<UserLogin>({
    id: "",
    password: "",
  });
  const [valid, setValid] = useState<LoginValid>({
    idValid: false,
    pwValid: false,
  });

  const handleOnChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setLoginInfo((cur): UserLogin => {
      const newInfo: UserLogin = { ...cur };
      newInfo[e.target.name as keyof UserLogin] = e.target.value; // 정석
      return newInfo;
    });
  };

  useEffect((): void => {
    if (loginInfo.id !== "") {
      setValid({
        ...valid,
        idValid: loginInfo.id.toLowerCase().match(reg) !== null ? true : false,
      });
    } else {
      setValid({
        ...valid,
        idValid: false,
      });
    }
  }, [loginInfo.id]);

  useEffect((): void => {
    if (loginInfo.password != "") {
      setValid({
        ...valid,
        pwValid: loginInfo.password.length >= 4 ? true : false,
      });
    } else {
      setValid({
        ...valid,
        pwValid: false,
      });
    }
  }, [loginInfo.password]);

  return (
    <>
      <LoginPage>
        <LoginBackground loginImg={loginImg}>
          <LoginForm>
            <InputBox>
              <h2 style={{ fontWeight: "bold" }}>아이디</h2>
              <LoginInput
                type="email"
                placeholder="이메일"
                name="id"
                value={loginInfo.id}
                onChange={handleOnChange}
              />
            </InputBox>
            {!valid.idValid && (
              <ValidWord>이메일 형식의 아이디가 아닙니다.</ValidWord>
            )}
            <InputBox>
              <h2 style={{ fontWeight: "bold" }}>비밀번호</h2>
              <LoginInput
                type="password"
                placeholder="비밀번호"
                name="password"
                value={loginInfo.password}
                onChange={handleOnChange}
              />
            </InputBox>
            {!valid.pwValid && (
              <ValidWord>비밀번호 4글자 이상 필요합니다.</ValidWord>
            )}
            <LoginButton disabled={!(valid.idValid && valid.pwValid)}>
              로그인
            </LoginButton>
          </LoginForm>
        </LoginBackground>
      </LoginPage>
      {/* <button onClick={() => setCount((c: number) => c + 1)}>+</button> */}
      <h2>{count2}</h2>
    </>
  );
}

export default Login;
