import { FC, RefObject, useState, useRef, ChangeEvent } from "react";
import { useModelParams } from "../../hooks/useModelParams";
import { Button } from "../../../../components/Button/Button";
import { useVoices } from "../../../../hooks/useVoices";

type ModelParamsProps = {
  isConnected: boolean;
  modal?: RefObject<HTMLDialogElement>,
} &  ReturnType<typeof useModelParams>;
export const ModelParams:FC<ModelParamsProps> = ({
  textTemperature,
  textTopk,
  audioTemperature,
  audioTopk,
  padMult,
  repetitionPenalty,
  repetitionPenaltyContext,
  setParams,
  resetParams,
  isConnected,
  textPrompt,
  voicePrompt,
  randomSeed,
  modal,
}) => {
  const { voices, uploadVoice, uploading } = useVoices();
  const [modalVoicePrompt, setModalVoicePrompt] = useState<string>(voicePrompt);
  const [modalTextPrompt, setModalTextPrompt] = useState<string>(textPrompt);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const filename = await uploadVoice(file);
    if (filename) {
      setModalVoicePrompt(filename);
    }
    e.target.value = "";
  };
  return (
    <div className=" p-2 mt-6 self-center flex flex-col items-center text-center">
        <table>
          <tbody>
            <tr>
              <td>Text Prompt:</td>
              <td className="w-12 text-center">{modalTextPrompt}</td>
              <td className="p-2"><input className="align-middle bg-white text-black border border-gray-300 rounded px-2 py-1" disabled={isConnected} type="text" id="text-prompt" name="text-prompt" value={modalTextPrompt} onChange={e => setModalTextPrompt(e.target.value)} /></td>
            </tr>
            <tr>
              <td>Voice Prompt:</td>
              <td className="w-12 text-center">{modalVoicePrompt}</td>
              <td className="p-2">
                <div className="flex gap-1 items-center">
                  <select className="align-middle bg-white text-black border border-gray-300 rounded px-2 py-1" disabled={isConnected} id="voice-prompt" name="voice-prompt" value={modalVoicePrompt} onChange={e => setModalVoicePrompt(e.target.value)}>
                    {voices.map((voice) => {
                      const isCustom = voice.endsWith('.wav');
                      const baseName = voice.replace(/\.(pt|wav)$/, '');
                      const displayName = baseName
                        .replace(/^NAT/, 'NATURAL_')
                        .replace(/^VAR/, 'VARIETY_');
                      return (
                        <option key={voice} value={voice}>
                          {isCustom ? `${displayName} (custom)` : displayName}
                        </option>
                      );
                    })}
                  </select>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".wav"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                  <button
                    type="button"
                    disabled={isConnected || uploading}
                    onClick={() => fileInputRef.current?.click()}
                    className="bg-white text-black border border-gray-300 rounded px-2 py-1 hover:bg-gray-100 disabled:opacity-50"
                    title="Upload custom voice (.wav)"
                  >
                    {uploading ? "..." : "\u2B06"}
                  </button>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
        <div>
          <Button onClick={resetParams} className="m-2">Reset</Button>
          <Button onClick={() => {
            console.log("Validating params");
            setParams({
            textTemperature,
            textTopk,
            audioTemperature,
            audioTopk,
            padMult,
            repetitionPenalty,
            repetitionPenaltyContext,
            textPrompt: modalTextPrompt,
            voicePrompt: modalVoicePrompt,
            randomSeed,
          });
          modal?.current?.close()
        }} className="m-2">Validate</Button>
        </div>
    </div>
  )
};
