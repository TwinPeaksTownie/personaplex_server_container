import { useEffect, useState } from "react";

export interface AudioDevice {
    deviceId: string;
    label: string;
}

export const useAudioDevices = () => {
    const [devices, setDevices] = useState<AudioDevice[]>([]);

    useEffect(() => {
        const getDevices = async () => {
            try {
                // Request permission first to get labels (otherwise labels might be empty)
                await navigator.mediaDevices.getUserMedia({ audio: true });

                const deviceInfos = await navigator.mediaDevices.enumerateDevices();
                const audioInputs = deviceInfos
                    .filter((d) => d.kind === "audioinput")
                    .map((d) => ({
                        deviceId: d.deviceId,
                        label: d.label || `Microphone ${d.deviceId.slice(0, 5)}...`,
                    }));
                setDevices(audioInputs);
            } catch (e) {
                console.error("Error enumerating devices:", e);
            }
        };

        getDevices();

        // Listen for device changes
        navigator.mediaDevices.addEventListener("devicechange", getDevices);
        return () => {
            navigator.mediaDevices.removeEventListener("devicechange", getDevices);
        };
    }, []);

    return devices;
};
