/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_ALLOW_DEV_ACTOR_HEADERS?: string;
  readonly PROD?: boolean;
  readonly DEV?: boolean;
  readonly MODE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
